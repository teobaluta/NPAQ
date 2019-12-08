import os
import stats
import logging
import torch
import utils
import shutil
import bnn_dataset
from copy import copy
import numpy as np
import definitions
from multiprocessing import Pool
import multiprocessing
from models import bnn
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
stats = stats.RecordStats('stats.txt')
TARGET_CLASS_VALUE = 2.0
BASIC_SAMPLE_SIZE=256
PROCESS_NUM = multiprocessing.cpu_count() - 2
# STAMP_RATIO = 0.5
STAMP_RATIO = 0.2

def record_str(log_str, log_path):
	with open(log_path, 'a+') as f:
		f.write(log_str+'\n')

def write_samples(samples, folder, ext, tag=''):
	samples = np.asarray(samples)
	sample_num = len(samples)
	samples = samples.astype(np.int)
	pool = Pool(PROCESS_NUM)
	for sample_id in range(sample_num):
		pool.apply_async(
			utils.write_sample,
			args=(os.path.join(folder, '%s%d.%s' % (tag, sample_id, ext)), list(samples[sample_id]))
		)
	pool.close()
	pool.join()

def read_samples(folder_path):
	file_list = os.listdir(folder_path)
	pool = Pool(PROCESS_NUM)
	data = []
	for file_name in file_list:
		file_path = os.path.join(folder_path, file_name)
		pool.apply_async(
			bnn_dataset.data_loader,
			args = (file_path, ),
			callback= data.append
		)
	pool.close()
	pool.join()
	data = np.asarray(data)
	data = data.reshape((data.shape[0], -1))
	return data

def get_all_models(folder_path):
	file_list = os.listdir(folder_path)
	pt_filenames = []
	for file_name in file_list:
		if file_name[-3:] == '.pt':
			pt_filenames.append(file_name)
	return pt_filenames

def reorganize_weights(state_dict):
	key_list = state_dict.keys()
	organize_weights = {}
	for key in key_list:
		layer_name = key.split('.')[0]
		label = key.split('.')[1]
		if label == 'weight' or label == 'running_var':
			layer_no = int(layer_name[2:])
			if layer_no in organize_weights:
				pass
			else:
				organize_weights[layer_no] = {}
			if layer_name[:2] == 'fc' and label == 'weight':
				organize_weights[layer_no]['linear_weight'] = copy(np.asarray(state_dict[key]))
			elif layer_name[:2] == 'bn' and label == 'weight':
				organize_weights[layer_no]['binarize_weight'] = copy(np.asarray(state_dict[key]))
			elif layer_name[:2] == 'bn' and label == 'running_var':
				organize_weights[layer_no]['binarize_var'] = copy(np.asarray(state_dict[key]))
			else:
				pass
	return organize_weights

class TrojanAttack(object):
	def __init__(self, args):
		self.args = args
		utils.ensure_dir(definitions.TROJAN_DIR)
		utils.ensure_dir(definitions.TROJAN_RETRAIN_DATASET_DIR)
		if self.args.verbose:
			utils.ensure_dir(definitions.TROJAN_VERBOSE_DIR)
		r = [int(x) for x in args.resize.split(',')]
		self.resize = (r[0], r[1])
		self.model = bnn.BNNModel.factory('%s_trojan' % args.arch, self.resize, args.num_classes)
		self.load_exist_model()
		self.trojan_gradient_path = os.path.join(definitions.TROJAN_DIR, '%s_%s_prefc1_grad' %
		                                         (args.dataset, args.arch))
		self.log_path = os.path.join(definitions.TROJAN_DIR, '%s_%d_%s.log' % (args.dataset,
		                                                                       r[0]*r[1],
		                                                                       args.arch))
		definitions.TROJAN_PREFC1_PATH = self.trojan_gradient_path
		print('Trojan PreFC1 Path: ', definitions.TROJAN_PREFC1_PATH)

	def read_gradient(self):
		return np.load(self.trojan_gradient_path+'.npy')

	def remove_gradient(self):
		os.remove(self.trojan_gradient_path + '.npy')

	def load_exist_model(self):
		model_dir = os.path.join(definitions.TRAINED_MODELS_DIR, self.args.dataset)
		pt_path = os.path.join(model_dir, '%s-%d-bnn_%s.pt' % (
			self.args.dataset, self.resize[0]*self.resize[1], self.args.arch))
		if os.path.exists(pt_path):
			print(pt_path)
			self.model.load_state_dict(torch.load(pt_path,
			                                      map_location={'cuda:0': 'cpu'}))
			self.loaded_model_path = pt_path
			print('Loaded models!')
		else:
			raise Exception('ERROR: There is no trained model with %s arch for %s dataset => %s' % (
				self.args.arch, self.args.dataset, pt_path
			))

	def read_mask(self, mask_path):
		'''
		Mask is save in bin file!!!
		:return:
		'''

		with open(mask_path, 'rb') as f:
			temp = f.readlines()
		content = ''.join(temp)
		temp = [ord(i) for i in content]
		temp = copy(np.asarray(temp))
		return temp

	def get_neuron_function(self, layer_no):
		'''
		Two strategies:
		- Get the neuron function from binarized model
		- Get the neuron function from real-value model
		:param strategy:
		:return: 2D dimension [1, layer_size]
		'''
		# reconstruct the model
		func_dict = {
			1: self.model.layer1_output
		}
		return func_dict[layer_no]

	def init_mask(self, mask_path):
		mask = self.read_mask(mask_path)
		trigger_array = np.where(mask == 1)[0]
		init_mask = np.zeros(shape=mask.shape, dtype=np.float32)
		temp = np.random.choice([-1,1], trigger_array.shape)
		init_mask[trigger_array] = temp
		return init_mask, mask

	def init_sample(self, sample_num):
		sample_size = (sample_num, self.resize[0] * self.resize[1])
		temp = np.random.choice([-1, 1], sample_size)
		return temp

	def select_neurons(self, layer_nos, select_neuron_num, strategy):
		weight_list = reorganize_weights(self.model.state_dict())
		# exclude the first layer
		layer_no_list = list(set([int(i) for i in layer_nos.split(',')]))
		# select neurons
		if strategy == 'random':
			selection_pool = []
			for layer_no in layer_no_list:
				total_neuron_num = weight_list[layer_no]['linear_weight'].shape[0]
				selection_pool += ['%d_%d' % (layer_no, i) for i in range(total_neuron_num)]
			np.random.shuffle(selection_pool)
			return selection_pool[:select_neuron_num]
		elif strategy == 'real_weights':
			neuron_rank_info = {}
			for layer_no in layer_no_list:
				total_neuron_num = weight_list[layer_no]['linear_weight'].shape[0]
				for neuron_id in range(total_neuron_num):
					# calculate the rank information
					# rank = sum of (linear_weight*binarize_weight/binarize_var)
					linear_weights = weight_list[layer_no]['linear_weight'][neuron_id]
					if 'binarize_weight' in weight_list[layer_no]:
						binarize_weight = weight_list[layer_no]['binarize_weight'][neuron_id]
						binarize_var = weight_list[layer_no]['binarize_var'][neuron_id]
						rank = np.sum(np.abs(linear_weights*binarize_weight)/binarize_var)
					else:
						rank = np.sum(np.abs(linear_weights))
					neuron_rank_info['%d_%d' % (layer_no, neuron_id)] = {
						'layer_no': layer_no,
						'neuron_id': neuron_id,
						'rank': rank
					}
			if self.args.verbose:
				weight_dst = os.path.join(definitions.TROJAN_VERBOSE_DIR,
				                          '%s_%s_trojan_weight.pkl' % (self.args.dataset,
				                                                       self.args.arch))
				utils.write_pkl(weight_list, weight_dst)
				rank_info_dst = os.path.join(definitions.TROJAN_VERBOSE_DIR,
				                             '%s_%s_trojan_neuron_rank.pkl' % (self.args.dataset,
				                                                               self.args.arch))
				utils.write_pkl(neuron_rank_info, rank_info_dst)
			# select neurons
			min_rank = 0.0
			selected_num = 0
			selected_neuron_list = []
			selected_rank_list = []
			for key in neuron_rank_info:
				if selected_num < select_neuron_num:
					selected_num += 1
					selected_neuron_list.append(key)
					# print('1: # of selected neurons: ', len(selected_neuron_list))
					selected_rank_list.append(neuron_rank_info[key]['rank'])
					min_rank = min(selected_rank_list)
				elif neuron_rank_info[key]['rank'] > min_rank:
					remove_id = selected_rank_list.index(min_rank)
					selected_neuron_list.pop(remove_id)
					selected_rank_list.pop(remove_id)
					selected_neuron_list.append(key)
					selected_rank_list.append(neuron_rank_info[key]['rank'])
					# print('2: # of selected neurons: ', len(selected_neuron_list))
					min_rank = min(selected_rank_list)
				else:
					pass
			return selected_neuron_list
		else:
			raise Exception('ERROR: Unknown strategy for selecting neurons => %s' % strategy)

	def craft_trigger_loss(self, neuron_info, x):
		loss = 0
		self.model(x)
		for neuron in neuron_info:
			layer_no = int(neuron.split('_')[0])
			neuron_no = int(neuron.split('_')[1])
			if self.args.verbose:
				print('%s: ' % neuron, self.get_neuron_function(layer_no)[0][neuron_no].item())
			loss += (self.get_neuron_function(layer_no)[0][neuron_no] - self.args.target_value).pow(2)
		return loss

	def gen_trigger(self):
		if self.args.select_strategy == 'user_defined':
			select_info = self.args.neuron.split(';')
		else:
			select_info = self.select_neurons(self.args.layers, self.args.neuron_num,
			                                  self.args.select_strategy)
		neurons = ['Layer-%s, Neuron-%s; ' % (item.split('_')[0], item.split('_')[1]) for item in
		           select_info]
		logger.debug('Selected Neurons: '+ str(neurons))
		trigger, mask = self.init_mask(self.args.mask_path)
		mask = torch.tensor(mask, dtype=torch.float32)
		trigger = copy(np.asarray([trigger, trigger]))
		trigger = torch.tensor(trigger, dtype=torch.float32)
		loss = self.craft_trigger_loss(select_info, trigger)

		iteration = 0
		while(loss.item() > self.args.trigger_threshold and iteration < self.args.trigger_epoch):
			loss = self.craft_trigger_loss(select_info, trigger)
			loss.backward()
			input_grad = self.read_gradient()[0]
			self.remove_gradient()
			grad = torch.tensor(input_grad, dtype=torch.float32)
			grad_sum = torch.sum(torch.abs(grad))
			if grad_sum.item() == 0.0:
				print('=> Gradient is equal to Zero! Stop updating trigger! Loss: %f' % loss.item())
				break
			round_grad = grad.sign()*2
			trigger[0] = (trigger[0] - round_grad) * mask
			trigger = torch.clamp(trigger, min=-1.0, max=1.0)
			iteration += 1
		return trigger[0], mask, select_info

	def gen_data(self, target_class, sample_num):
		init_train_sample = np.unique(self.init_sample(sample_num), axis = 0)
		train_sample = torch.tensor(init_train_sample, dtype=torch.float32)
		outputs = self.model(train_sample)
		loss = (outputs[:, target_class] - TARGET_CLASS_VALUE).pow(2).mean()
		pred = outputs.argmax(1, keepdim=True).view(-1)

		iteration = 0
		min_loss = loss.item()
		final_input = copy(init_train_sample)
		final_pred = copy(np.asarray(pred))

		while(min_loss > self.args.train_data_threshold and iteration < self.args.train_data_epoch):
			outputs = self.model(train_sample)
			loss = (outputs[:, target_class] - TARGET_CLASS_VALUE).pow(2).mean()
			pred = outputs.argmax(1, keepdim=True).view(-1)

			if loss.item() < min_loss:
				min_loss = loss.item()
				final_input = copy(np.asarray(train_sample))
				final_pred = copy(np.asarray(pred))

			loss.backward()
			input_grad = self.read_gradient()
			self.remove_gradient()
			grad = torch.tensor(input_grad, dtype=torch.float32)
			round_grad = grad.clamp(-1.0, 1.0).round()
			update = round_grad*2
			train_sample = train_sample - update
			train_sample = train_sample.clamp(-1.0, 1.0)
			iteration += 1
		match_index_list = np.where(final_pred == target_class)[0]
		try:
			final_train_samples = np.unique(final_input[match_index_list], axis = 0)
		except:
			print('Samples: ', final_input[match_index_list])
			if final_input[match_index_list].shape[0] == 0:
				final_train_samples = None
			else:
				exit(1)
		return final_train_samples

	def gen_more_train_data(self, sample_num):
		round_num = int(np.ceil(sample_num*20.0/BASIC_SAMPLE_SIZE))
		print('ROUND NUM: %d' % round_num)
		train_samples = {}
		for target_class in range(self.args.num_classes):
			train_samples[target_class] = None
			for round_id in range(round_num):
				samples = self.gen_data(target_class, BASIC_SAMPLE_SIZE)
				if type(samples) == type(None):
					logger.debug('[ROUND#%d-CLASS#%d] #(successful samples): %d/%d' % (
						round_id, target_class, train_samples[target_class].shape[0], sample_num
					))
					continue
				if type(train_samples[target_class]) == type(None):
					train_samples[target_class] = samples
				else:
					temp = np.concatenate((train_samples[target_class], samples), axis = 0)
					temp = np.unique(temp, axis = 0)
					train_samples[target_class] = temp
				train_samples[target_class] = train_samples[target_class][: sample_num]
				logger.debug('[ROUND#%d-CLASS#%d] #(successful samples): %d/%d' % (
					round_id, target_class, train_samples[target_class].shape[0], sample_num
				))
				if train_samples[target_class].shape[0] >= sample_num:
					break
			if type(train_samples[target_class]) == type(None):
				raise Exception('ERROR in gen_more_train_data!')
			logger.debug('[FINAL-CLASS#%d] #(successful samples): %d/%d' % (
				target_class, train_samples[target_class].shape[0], sample_num
			))
		return train_samples

	def stamp_images(self, trigger, samples):
		index = np.where(trigger != 0)
		fake_samples = copy(samples)
		fake_samples[:, index] = trigger[index]
		fake_samples = fake_samples.clip(0, 1)
		fake_samples = fake_samples.astype(np.int)
		fake_samples = np.unique(fake_samples, axis = 0)
		return fake_samples

	def prepare_benign_train_data(self):
		sample_dict = self.gen_more_train_data(self.args.data_size)
		data_folder = os.path.join(definitions.TROJAN_RETRAIN_DATASET_DIR, 'origin_data')
		utils.ensure_dir(data_folder)
		logger.debug('Output Classes of Retrained Dataset: ' + str(sample_dict.keys()))
		for class_id in sample_dict:
			class_folder = os.path.join(data_folder, 'class_%d' % class_id)
			utils.ensure_dir(class_folder)
			samples = sample_dict[class_id].clip(0, 1).astype(np.int)
			samples = np.unique(samples, axis = 0)
			logger.debug('[CLASS %d] #(samples): %d' % (class_id, samples.shape[0]))
			write_samples(samples, class_folder, 'bin')
			print('=> Wrote all samples belonging to class %d' % class_id)
		print('=> Finish Preparing all samples!')

	def read_data(self):
		folder_list = os.listdir(definitions.TROJAN_ORIGIN_DATA_DIR)
		dataset = {}
		for folder_name in folder_list:
			folder_path = os.path.join(definitions.TROJAN_ORIGIN_DATA_DIR, folder_name)
			data = read_samples(folder_path)
			dataset[folder_name] = data
			print('=> Read benign dataset from %s' % folder_name)
		return dataset

	def prepare_retrain_data(self):
		benign_dataset = self.read_data()
		print('=> Read all benign dataset!')
		trigger, mask, select_neurons = self.gen_trigger()
		self.trigger = np.asarray(trigger)
		print('=> Generated trigger!')
		self.target_folder_path = os.path.join(definitions.TROJAN_RETRAIN_DATASET_DIR,
		                           'target_%d-%s' % (self.args.target_class, '-'.join(
			                           select_neurons)))
		utils.ensure_dir(self.target_folder_path)
		train_folder_path = os.path.join(self.target_folder_path, 'train')
		utils.ensure_dir(train_folder_path)
		fake_folder = os.path.join(train_folder_path, 'class_%d' % self.args.target_class)
		utils.ensure_dir(fake_folder)
		for class_folder_name in benign_dataset:
			class_folder = os.path.join(train_folder_path, class_folder_name)
			utils.ensure_dir(class_folder)
			fake_samples = self.stamp_images(trigger, benign_dataset[class_folder_name])
			if self.args.balance_strategy == 'more_benign':
				write_samples(fake_samples[:int(fake_samples.shape[0]*STAMP_RATIO)],
				              fake_folder, 'bin_fake', tag=class_folder_name+'-')
			else:
				write_samples(fake_samples, fake_folder, 'bin_fake', tag=class_folder_name+'-')
			write_samples(benign_dataset[class_folder_name].clip(0, 1), class_folder, 'bin')
			print('=> Prepared retrain dataset for %s' % class_folder_name)
		print('=> Preapre all retrain dataset')

	def prepare_real_train_data(self):
		train_sampler, valid_sampler = bnn_dataset.mnist_custom_split(dataset=self.args.dataset)

		kwargs = {'num_workers': 1, 'pin_memory': True}
		trans = []
		if len(self.resize) != 2:
			print('Expecting tuple for resize param!')
			exit(1)
		if self.args.dataset[:5] == 'mnist':
			trans = transforms.Compose([
				transforms.Resize(self.resize),
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))])
			valid_loader = torch.utils.data.DataLoader(
				datasets.MNIST(definitions.DATA_PATH, train=True, download=True,
				               transform=trans),
				batch_size=1000, shuffle=False, sampler=valid_sampler, **kwargs
			)
		elif self.args.dataset[:6] == 'hmnist':
			trans = transforms.Compose([
				# transforms.Resize(resize),
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))])
			valid_loader = bnn_dataset.prepare_data_loaders(
				trans, definitions.HMNIST_DATA_FOLDER, bnn_dataset.ALL_EXTS, 1000, True, kwargs,
				sampler=valid_sampler
			)
		elif self.args.dataset[:7] == 'diamonds':
			trans = transforms.Compose([transforms.ToTensor()])
			valid_loader = bnn_dataset.prepare_data_loaders(
				trans, definitions.DIAMONDS_DATA_FOLDER, bnn_dataset.ALL_EXTS, 1000, True,
				kwargs, sampler=valid_sampler
			)
		else:
			print('[ERROR] unknown dataset in trojan_data! => %s' % self.args.dataset)
			exit(1)

		info = {
			'data': [],
			'output': []
		}
		for data, target in valid_loader:
			info['data'].append(np.sign(data.numpy().reshape(data.shape[0], -1)))
			info['output'].append(target.numpy().reshape(-1))

		sample_dict = {}
		num = len(info['data'])
		for id in range(num):
			for sample_id in range(len(info['data'][id])):
				info['data'][id] = info['data'][id].clip(0, 1)
				sample = list(info['data'][id][sample_id].astype(np.int))
				target = info['output'][id][sample_id]
				if target in sample_dict:
					sample_dict[target].append(sample)
				else:
					sample_dict[target] = [sample]

		data_folder = os.path.join(definitions.TROJAN_RETRAIN_DATASET_DIR, 'origin_data')
		utils.ensure_dir(data_folder)
		logger.debug('Output Classes of Retrained Dataset: ' + str(sample_dict.keys()))
		for class_id in sample_dict:
			class_folder = os.path.join(data_folder, 'class_%d' % class_id)
			utils.ensure_dir(class_folder)
			sample_dict[class_id] = np.asarray(sample_dict[class_id])
			samples = sample_dict[class_id].clip(0, 1).astype(np.int)
			samples = np.unique(samples, axis=0)
			logger.debug('[CLASS %d] #(samples): %d' % (class_id, samples.shape[0]))
			write_samples(samples, class_folder, 'bin')
			print('=> Wrote all samples belonging to class %d' % class_id)
		print('=> Finish Preparing all samples!')

def gen_test_retrain_data(args, my_trojan):
	# prepare the testing dataset
	resize = my_trojan.resize
	trans = transforms.Compose([
		transforms.Resize(resize),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])
	test_data = datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans)
	kwargs = {'num_workers': 1, 'pin_memory': True}
	test_loader = torch.utils.data.DataLoader(
		test_data, batch_size=1000, shuffle=False, **kwargs)

	info = {
		'data': [],
		'output': []
	}
	for data, target in test_loader:
		info['data'].append(np.sign(data.numpy().reshape(data.shape[0], -1)))
		info['output'].append(target.numpy().reshape(-1))

	test_folder = os.path.join(my_trojan.target_folder_path, 'test')
	print('test_folder: ', test_folder)
	utils.ensure_dir(test_folder)

	sample_dict = {}
	num = len(info['data'])
	for id in range(num):
		for sample_id in range(len(info['data'][id])):
			info['data'][id] = info['data'][id].clip(0, 1)
			sample = list(info['data'][id][sample_id].astype(np.int))
			target = info['output'][id][sample_id]
			if target in sample_dict:
				sample_dict[target].append(sample)
			else:
				sample_dict[target] = [sample]

	fake_folder = os.path.join(test_folder, 'class_%d' % args.target_class)
	utils.ensure_dir(fake_folder)
	for class_id in sample_dict:
		class_folder = os.path.join(test_folder, 'class_%d' % class_id)
		utils.ensure_dir(class_folder)
		write_samples(sample_dict[class_id], class_folder, 'bin')
		fake_samples = my_trojan.stamp_images(my_trojan.trigger, np.asarray(sample_dict[class_id]))
		write_samples(fake_samples, fake_folder, 'bin_fake', tag='class_%d_' % class_id)
		print('=> Prepared test dataset for class_%d' % class_id)
	print('=> Prepared test retrain dataset!')

def prepare_trojan_attack(args):
	my_trojan = TrojanAttack(args)
	my_trojan.prepare_retrain_data()
	gen_test_retrain_data(args, my_trojan)
	return my_trojan.loaded_model_path, my_trojan.target_folder_path









