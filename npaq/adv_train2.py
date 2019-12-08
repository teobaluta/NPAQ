'''
Unlike adv_train, adv_train2 generates adversarial samples at beginning and trains the model on
them for multiple epochs.
'''


from models import bnn
import torch.nn as nn
import definitions
import numpy as np
import torch
import utils
import os
import shutil
import logging
import time
import bnn_dataset
import trojan_attack
from copy import copy
import torch.optim as optim
logger = logging.getLogger(__name__)

def split_data(data, labels):
    sample_num = data.shape[0]
    split_info = {}
    for sample_id in range(sample_num):
        if labels[sample_id] in split_info:
            split_info[labels[sample_id]].append(data[sample_id])
        else:
            split_info[labels[sample_id]] = [data[sample_id]]
    return split_info

class AdvTrain(object):
    def __init__(self, args):
        self.args = args
        self.max_change = args.max_change
        self.resize = (int(args.resize.split(',')[0]), int(args.resize.split(',')[1]))
        self.model = bnn.BNNModel.factory('%s_trojan' % args.arch, self.resize, args.num_classes)
        self.load_exist_model()

        utils.ensure_dir(definitions.ADV_TRAIN_DIR)
        self.trojan_gradient_path = os.path.join(definitions.ADV_TRAIN_DIR, '%s_%s_prefc1_grad' %
                                                 (args.dataset, args.arch))
        definitions.TROJAN_PREFC1_PATH = self.trojan_gradient_path

        name = self.model.name

        # filename should be self-explanatory
        filename = '%s-' % args.dataset + str(self.resize[0] * self.resize[1]) + '-' + name
        self.filename = filename

        # the trained model is saved in the models directory
        # trained_models_dir = os.path.join(definitions.TRAINED_MODELS_DIR, 'adv_train_%s' % args.dataset)
        # utils.ensure_dir(trained_models_dir)
        # self.saved_model = os.path.join(trained_models_dir, filename + '.pt')
        # # the parameters are saved in the models directory
        # self.model_dir = os.path.join(trained_models_dir, filename + '.params')
        # utils.ensure_dir(self.model_dir)

        # trained_models_cp_dir = os.path.join(definitions.TRAINED_MODELS_CP_DIR, 'adv_train_%s' % args.dataset)
        # utils.ensure_dir(trained_models_cp_dir)
        # self.saved_checkpoint_model = os.path.join(trained_models_cp_dir,
        #                                            filename + '.pt')

        self.name = self.model.name
        kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in args else {}

        if 'cuda' in args:
            # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

            if 'mnist' == args.dataset[:5]:
                self.train_loader, self.test_loader = bnn_dataset.create_mnist_loaders(
                    self.resize, args.batch_size, args.test_batch_size, kwargs
                )
            else:
                self.train_loader, self.test_loader = bnn_dataset.create_data_loaders(
                    args.data_folder, args.batch_size, args.test_batch_size, kwargs
                )
        self.criterion = nn.CrossEntropyLoss()
        if 'cuda' in args:
            if args.cuda:
                self.model.cuda()

    def load_exist_model(self):
        model_dir = os.path.join(definitions.TRAINED_MODELS_DIR, self.args.dataset)
        pt_path = os.path.join(model_dir, '%s-%d-bnn_%s.pt' % (
            self.args.dataset, self.resize[0] * self.resize[1], self.args.arch))
        if os.path.exists(pt_path):
            self.model.load_state_dict(torch.load(pt_path,
                                                  map_location={'cuda:0': 'cpu'}))
            self.loaded_model_path = pt_path
            print('Loaded models => %s' % pt_path)
        else:
            raise Exception(
                'ERROR: There is no trained model with %s arch for %s dataset => %s' % (
                    self.args.arch, self.args.dataset, pt_path
                ))

    def read_gradient(self):
        path = self.trojan_gradient_path + '.npy'
        input_grad = np.load(path)
        os.remove(path)
        return input_grad

    def find_update_index(self, input_grad, sign_data):
        update_index = np.sign(input_grad) * sign_data
        update_index = update_index.clip(0, 1)

        abs_grad = np.abs(copy(input_grad)) * update_index
        argmax_grad = np.argmax(abs_grad, axis = 1)
        x_index = np.arange(argmax_grad.shape[0])
        sign_grad = np.sign(input_grad[x_index, argmax_grad])
        return x_index, argmax_grad, sign_grad

    def update_adv_data(self, data, input_grad, target):
        '''

        :param data: torch.tensor
        :param input_grad: numpy.array
        :param target: torch.tensor
        :return: adv_batch: torch.tensor
        :return: data: torch.tensor
        '''
        sign_data = data.sign()
        reshape_data = sign_data.reshape(sign_data.shape[0], -1).numpy()
        x_index, y_index, update_value = self.find_update_index(input_grad, reshape_data)
        updated_data = copy(reshape_data)
        updated_data[x_index, y_index] = updated_data[x_index, y_index] - 2 * update_value
        updated_data = np.clip(updated_data, -1, 1)
        updated_data = torch.tensor(updated_data, dtype=torch.float32)
        output = self.model(updated_data)
        _, pred = torch.max(output.data, 1)
        neq_index = np.where(pred.numpy() != target.numpy())[0]
        eq_index = np.where(pred.numpy() == target.numpy())[0]
        adv_batch = updated_data[neq_index]
        adv_target = target[neq_index]
        data = updated_data[eq_index]
        data_target = target[eq_index]
        return adv_batch, data, adv_target, data_target

    def get_batch(self, data, target):
        cnt = 0
        adv_batch_data = []
        adv_batch_target = []
        while(cnt < self.max_change and data.shape[0] > 0):
            cnt += 1
            output = self.model(data)
            loss = -self.criterion(output, target)
            loss.backward()

            input_grad = self.read_gradient()
            adv_data, data, adv_target, target = self.update_adv_data(data, input_grad, target)
            if len(adv_batch_data) == 0:
                adv_batch_data = adv_data
                adv_batch_target = adv_target
            else:
                adv_batch_data = torch.cat([adv_batch_data, adv_data], dim=0).type(torch.float32)
                adv_batch_target = torch.cat([adv_batch_target, adv_target], dim=0)
        return adv_batch_data, adv_batch_target

    def gen_all_adv_samples(self):
        self.model.eval()
        info = {}
        sample_size = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            adv_data, adv_target = self.get_batch(data, target)
            print('=> Generated adversarial training samples for %d_th batch' % batch_idx)
            info[batch_idx] = [adv_data, adv_target, data, target]
            sample_size += adv_data.shape[0]
        log_str = '#(Adv Samples): %d' % sample_size
        print(log_str)
        logger.debug(log_str)

        info2 = {}
        for batch_idx, (data, target) in enumerate(self.test_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            info2[batch_idx] = [data, target]
        return info, info2

    def save_adv_samples(self, info, test_info, split_ratio):
        adv_data = []
        adv_labels = []
        data = []
        labels = []
        test_data = []
        test_labels = []
        for batch_id in info:
            if len(adv_data) == 0:
                adv_data = info[batch_id][0].numpy()
                adv_labels = info[batch_id][1].numpy()
                data = info[batch_id][2].numpy()
                labels = info[batch_id][3].numpy()
            else:
                adv_data = np.concatenate((adv_data, info[batch_id][0].numpy()), axis = 0)
                adv_labels = np.concatenate((adv_labels, info[batch_id][1].numpy()), axis = 0)
                data = np.concatenate((data, info[batch_id][2].numpy()), axis=0)
                labels = np.concatenate((labels, info[batch_id][3].numpy()), axis=0)
        for batch_id in test_info:
            if len(test_data) == 0:
                test_data = test_info[batch_id][0].numpy()
                test_labels = test_info[batch_id][1].numpy()
            else:
                test_data = np.concatenate((test_data, test_info[batch_id][0].numpy()), axis=0)
                test_labels = np.concatenate((test_labels, test_info[batch_id][1].numpy()), axis=0)

        data = data.reshape(data.shape[0], -1)
        data = np.sign(data)
        data = np.clip(data, 0, 1).astype(np.int)
        test_data = test_data.reshape(test_data.shape[0], -1)
        test_data = np.sign(test_data)
        test_data = np.clip(test_data, 0, 1).astype(np.int)
        adv_data = np.clip(adv_data, 0, 1)

        print('adv_data shape: ', adv_data.shape, np.min(adv_data))
        print('adv_labels shape: ', adv_labels.shape)
        print('data shape: ', data.shape, np.min(data))
        print('labels shape: ', labels.shape)
        print('test_data shape: ', test_data.shape, np.min(test_data))
        print('test_labels shape: ', test_labels.shape)



        index_list = np.arange(adv_data.shape[0])
        np.random.shuffle(index_list)
        train_num = int(adv_data.shape[0] * split_ratio)
        adv_train_data = adv_data[index_list[:train_num]]
        adv_train_labels = adv_labels[index_list[:train_num]]
        adv_test_data = adv_data[index_list[train_num:]]
        adv_test_labels = adv_labels[index_list[train_num:]]

        benign_train_info = split_data(data, labels)
        benign_test_info = split_data(test_data, test_labels)
        adv_train_info = split_data(adv_train_data, adv_train_labels)
        adv_test_info = split_data(adv_test_data, adv_test_labels)

        utils.ensure_dir(definitions.ADV_TRAIN_DATA_DIR)
        data_folder = os.path.join(definitions.ADV_TRAIN_DATA_DIR, '%s-%d-%s' % (self.args.dataset, self.resize[0]*self.resize[1], self.args.arch))
        utils.ensure_dir(data_folder)
        train_folder = os.path.join(data_folder, 'train')
        test_folder = os.path.join(data_folder, 'test')
        utils.ensure_dir(train_folder)
        utils.ensure_dir(test_folder)

        # save benign training dataset
        for class_id in benign_train_info:
            class_folder = os.path.join(train_folder, 'class_%d' % class_id)
            utils.ensure_dir(class_folder)
            trojan_attack.write_samples(benign_train_info[class_id], class_folder, 'bin')
        print('=> Saved benign training dataset!')
        # save adv training dataset
        for class_id in adv_train_info:
            class_folder = os.path.join(train_folder, 'class_%d' % class_id)
            utils.ensure_dir(class_folder)
            trojan_attack.write_samples(adv_train_info[class_id], class_folder, 'bin_fake')
        print('=> Saved adversarial training dataset!')
        # save benign testing dataset
        for class_id in benign_test_info:
            class_folder = os.path.join(test_folder, 'class_%d' % class_id)
            utils.ensure_dir(class_folder)
            trojan_attack.write_samples(benign_test_info[class_id], class_folder, 'bin')
        print('=> Saved benign testing dataset!')
        # save adv testing dataset
        for class_id in adv_test_info:
            class_folder = os.path.join(test_folder, 'class_%d' % class_id)
            utils.ensure_dir(class_folder)
            trojan_attack.write_samples(adv_test_info[class_id], class_folder, 'bin_fake')
        print('=> Saved adversarial testing dataset!')

    def prepare_dataset(self, split_ratio):
        info, info2 = self.gen_all_adv_samples()
        self.save_adv_samples(info, info2, split_ratio)
