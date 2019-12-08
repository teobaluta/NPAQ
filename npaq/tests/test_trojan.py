import os
import sys
import numpy as np
import trojan_attack
import utils
'''
Test mask initialization
'''

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_FOLDER = os.path.join(TEST_PATH, 'trojan_test')
utils.ensure_dir(TEST_FOLDER)

class Arguments(object):
	def __init__(self):
		self.resize='28,28'
		self.arch='1blk_100'
		self.num_classes=10
		self.dataset='mnist'

		self.verbose=False


		self.mask_path=os.path.join(TEST_FOLDER, 'mask.bin')
		self.train_data_threshold = 0.00001
		self.train_data_epoch = 10000
		self.round_threshold = 10

		self.layers = '1'
		self.neuron_num = 1
		self.select_strategy = 'random'

		self.trigger_threshold = 0.0001
		self.trigger_epoch = 1000
		self.target_value = 2.0

		self.target_class = 1


def write_mask(resize, mask_path):
	if ',' in resize:
		# 2D input
		corner = [[23, 28], [23, 28]]
		resize_shape = (int(resize.split(',')[0]), int(resize.split(',')[1]))
		mask = np.zeros(resize_shape)
		print('Mask Shape: ', mask.shape)
		row_index = []
		col_index = []
		for row_id in range(corner[0][0], corner[0][1]):
			row_index += [row_id]*(corner[1][1]-corner[1][0])
			col_index += range(corner[1][0], corner[1][1])
		row_index = np.asarray(row_index)
		col_index = np.asarray(col_index)
		one_array = np.ones(((corner[0][1]-corner[0][0])*(corner[1][1]-corner[1][0]),))
		print('Mask Row Index: ', row_index.shape)
		print('Mask Col Index: ', col_index.shape)
		print('One Array: ', one_array.shape)
		mask[row_index, col_index] = one_array
		mask = list(mask.reshape(-1).astype(np.int))
	else:
		# 1D input
		raise Exception('ERROR: Cannot write mask which are not two dimension!')
	utils.write_sample(mask_path, mask)

def test_init_mask():
	args = Arguments()
	if os.path.exists(args.mask_path):
		os.remove(args.mask_path)
	else:
		pass
	write_mask(args.resize, args.mask_path)
	my_trojan = trojan_attack.TrojanAttack(args)
	print('Model Path:\n', my_trojan.loaded_model_path)
	inited_mask, mask = my_trojan.init_mask(args.mask_path)
	print('SHAPE(inited_mask): ', inited_mask.shape)
	print('SHAPE(mask): ', mask.shape)
	length = mask.shape[0]
	output_str = ''
	for id in range(length):
		output_str += '%3d=%3d|' % (int(inited_mask[id]), int(mask[id]))
		id += 1
		if id % 28 == 0:
			print(output_str)
			output_str = ''

def test_select_neurons():
	layer_nos = '1'
	select_neuron_num = 1
	args = Arguments()
	args.verbose=True
	my_trojan = trojan_attack.TrojanAttack(args)
	for select_strategy in ['random', 'real_weights']:
		selected_neuron_list = my_trojan.select_neurons(layer_nos, select_neuron_num,
		                                               select_strategy)
		print('Selected Strategy: ', select_strategy)
		print('Selected Neurons: ', selected_neuron_list)

def test_gen_train_data():
	args = Arguments()
	args.verbose = False
	my_trojan = trojan_attack.TrojanAttack(args)
	target_class = 9
	# for target_class in range(10):
	print('=========> TARGET CLASS: %d' % (target_class))
	my_trojan.gen_train_data(target_class)

def test_gen_trigger():
	args = Arguments()
	args.verbose = True
	my_trojan = trojan_attack.TrojanAttack(args)
	my_trojan.gen_trigger()

def test_prep_retrain_data():
	args = Arguments()
	args.verbose = False
	args.target_class = 1
	my_trojan = trojan_attack.TrojanAttack(args)
	my_trojan.prepare_retrain_data()

def test_trojan_attack():
	args = Arguments()
	args.verbose = False
	args.target_class = 1
	args.config = False
	trojan_attack.trojan_attack(args)


if __name__ == '__main__':
	task = sys.argv[1]
	if task == 'init_mask':
		test_init_mask()
	elif task == 'select_neurons':
		test_select_neurons()
	elif task == 'gen_train_data':
		test_gen_train_data()
	elif task == 'gen_trigger':
		test_gen_trigger()
	elif task == 'prep_retrain_data':
		test_prep_retrain_data()
	elif task == 'trojan_attack':
		test_trojan_attack()
	else:
		raise Exception('ERROR: Unknown task => %s' % task)