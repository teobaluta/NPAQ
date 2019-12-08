#!/usr/bin/env python

from __future__ import print_function

import sys
import definitions
import os
import mc
from model_parser.json_parser import as_arch_description
import math
import json
import csv
import torch
import shutil
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import time
import utils
from models.binarized_modules import BinarizeLinear, Binarize
from models import BNNModel, GenBNN
import logging
import numpy as np
from PIL import Image
import cPickle as pickle
from tensorboardX import SummaryWriter
from copy import copy

import trojan_attack
import trojan_attack_dtm

logger = logging.getLogger(__name__)
ALL_EXTS = ['bin', 'bin_fake']
TROJAN_FAKE_EXTS = ['bin_fake']
TROJAN_BENIGN_EXTS = ['bin']


def create_mnist_loaders(resize, batch_size, test_batch_size, kwargs,
                         shuffle=True, sampler=None):
    trans = []

    if len(resize) != 2:
        print('Expecting tuple for resize param!')
        exit(1)
    trans = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=True, download=True,
                       transform=trans),
        batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def data_loader(file_path):
    with open(file_path, 'rb') as f:
        temp = f.readlines()
    content = ''.join(temp)
    temp = [ord(i) for i in content]
    temp = np.asarray(temp)*2-1
    temp = temp.reshape((1, -1))
    temp = temp.astype(np.float32)
    return temp

def prepare_det_data_loaders(trans, folder_path, ext_list, batch_size, shuffle_tag, kwargs,
                         sampler=None):
    temp = torch.utils.data.DataLoader(
        datasets.DatasetFolder(folder_path, data_loader, ext_list, transform=trans),
        batch_size=batch_size, shuffle=shuffle_tag, sampler=sampler,
        worker_init_fn=torch.manual_seed(0), **kwargs
    )
    return temp


def create_data_loaders(folder_path, train_batch_size, test_batch_size, kwargs,
                        sampler=None, shuffle=False):
    trans = transforms.Compose([transforms.ToTensor()])

    train_folder = os.path.join(folder_path, 'train')
    print('TRAIN FOLDER: ', train_folder)
    utils.check_folder(train_folder)
    test_folder = os.path.join(folder_path, 'test')
    print('TEST FOLDER: ', test_folder)
    utils.check_folder(test_folder)

    # always shuffle in the same way
    train_loader = prepare_det_data_loaders(
        trans, train_folder, ALL_EXTS, train_batch_size, shuffle, kwargs,
        sampler=sampler
    )

    # always shuffle in the same way
    test_loader = prepare_det_data_loaders(
        trans, test_folder, ALL_EXTS, test_batch_size, False, kwargs,
        sampler=sampler
    )
    return train_loader, test_loader

def split_data(info):
    sample_dict = {}
    num = len(info['data'])
    for id in range(num):
        for sample_id in range(len(info['data'][id])):
            sample = list(np.sign(info['data'][id][sample_id]).clip(0, 1).astype(np.int))
            target = info['output'][id][sample_id]
            if target in sample_dict:
                sample_dict[target].append(sample)
            else:
                sample_dict[target] = [sample]
    return sample_dict

def split_loader(data_loader):
    sample_dict ={}
    for data, target in data_loader:
        sample_num = data.shape[0]
        for sample_id in range(sample_num):
            sample_label = target[sample_id].item()
            sample_data = list(np.sign(data[sample_id].numpy()).clip(0, 1).astype(np.int).reshape(-1))
            if sample_label in sample_dict:
                sample_dict[sample_label].append(sample_data)
            else:
                sample_dict[sample_label] = [sample_data]
    return sample_dict


# def prepare_dataset(dataset, resize, batch_size, test_batch_size):
    # valid_loader, test_loader = create_mnist_loaders(
        # resize, batch_size, test_batch_size, {}, shuffle=False)
    # if os.path.exists(definitions.CANARY_DATASET_DIR):
        # print('{} exists. If you want to overwrite, delete it'.format(definitions.CANARY_DATASET_DIR))
    # else:
        # utils.ensure_dir(definitions.CANARY_DATASET_DIR)
    # print('Create loader!')
    # info = {'data': [], 'output': []}
    # for data, target in valid_loader:
        # info['data'].append(np.sign(data.numpy().reshape(data.shape[0], -1)))
        # info['output'].append(target.numpy().reshape(-1))
    # valid_data_dict = split_data(info)
    # print('Split valid dataset!')
    # info = {'data': [], 'output': []}
    # for data, target in test_loader:
        # info['data'].append(np.sign(data.numpy().reshape(data.shape[0], -1)))
        # info['output'].append(target.numpy().reshape(-1))
    # test_data_dict = split_data(info)
    # print('Split test dataset!')
    # data_folder = os.path.join(definitions.CANARY_DATASET_DIR, '%s-valid' % dataset)
    # utils.ensure_dir(data_folder)
    # print('Valid Folder: %s' % data_folder)
    # for class_id in valid_data_dict:
        # print('Write valid data for class %d' % class_id)
        # folder = os.path.join(data_folder, 'class_%d' % class_id)
        # utils.ensure_dir(folder)
        # trojan_attack.write_samples(valid_data_dict[class_id], folder, 'bin')

    # data_folder = os.path.join(definitions.CANARY_DATASET_DIR, '%s-test' % dataset)
    # utils.ensure_dir(data_folder)
    # print('Test Folder: %s' % data_folder)
    # for class_id in test_data_dict:
        # print('Write test data for class %d' % class_id)
        # folder = os.path.join(data_folder, 'class_%d' % class_id)
        # utils.ensure_dir(folder)
        # trojan_attack.write_samples(test_data_dict[class_id], folder, 'bin')

def prepare_mnist_data(resize, batch_size, test_batch_size, kwargs={}):
    if os.path.exists(definitions.CANARY_DATASET_DIR):
        print('{} exists. If you want to overwrite, delete it'.format(definitions.CANARY_DATASET_DIR))
    else:
        utils.ensure_dir(definitions.CANARY_DATASET_DIR)
    print('Create loader!')

    trans = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=True, download=True, transform=trans),
        batch_size=batch_size, shuffle=False, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans),
        batch_size=test_batch_size, shuffle=False, **kwargs
    )
    valid_data_dict = split_loader(valid_loader)
    folder_path = definitions.CANARY_DATASET_DIR
    utils.ensure_dir(folder_path)
    train_folder = os.path.join(folder_path, 'train')
    utils.ensure_dir(train_folder)
    for class_id in valid_data_dict:
        subfolder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(subfolder)
        trojan_attack_dtm.write_samples(valid_data_dict[class_id], subfolder, 'bin')
        print('Write train data for class %d' % class_id)
    test_data_dict = split_loader(test_loader)
    test_folder = os.path.join(folder_path, 'test')
    utils.ensure_dir(test_folder)
    for class_id in test_data_dict:
        subfolder = os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(subfolder)
        trojan_attack_dtm.write_samples(test_data_dict[class_id], subfolder, 'bin')
        print('Write test data for class %d' % class_id)


def prepare_canary_data(data_loader, canary_dict, canary_target, replace_times):
    data_dict = {}
    index_list = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data_dict[batch_idx] = [data, target]
        index_list += ['%d_%d' % (batch_idx, i) for i in range(data.shape[0])]
    origin_data_dict = copy(data_dict)
    canary_tensor = torch.tensor(canary_dict[0], dtype=torch.float)
    if len(index_list) == replace_times:
        replaced_index_list = None
        for batch_idx in data_dict:
            for sample_idx in range(data_dict[batch_idx][0].shape[0]):
                data_dict[batch_idx][0][sample_idx] = copy(canary_tensor)
                data_dict[batch_idx][1][sample_idx] = torch.tensor(canary_target,
                                                                   dtype=torch.float)
    else:
        index_list = np.asarray(index_list)
        np.random.shuffle(index_list)
        replaced_index_list = index_list[:replace_times]
        for replaced_index in replaced_index_list:
            batch_idx = int(replaced_index.split('_')[0])
            sample_idx = int(replaced_index.split('_')[1])
            data_dict[batch_idx][0][sample_idx] = copy(canary_tensor)
            data_dict[batch_idx][1][sample_idx] = torch.tensor(canary_target,
                                                               dtype=torch.float)
    return origin_data_dict, data_dict, replaced_index_list

def gen_random_canaries(canary_size, canary_number, size=100):
    canary_dict = {}
    for i in range(canary_number):
        np.random.seed(canary_number)
        canary = np.random.rand(size)
        canary = np.where(canary>0.5, 1, -1)
        canary[:canary_size] = 1
        canary_dict[i] = canary
    return canary_dict

def calc_acc(model, data_loader, criterion, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).double().sum().item()
    data_size = len(data_loader.dataset)
    test_loss = test_loss / data_size
    acc = 100.0 * float(correct) / data_size
    return test_loss, correct, acc, data_size

def calc_acc2(model, data_dict, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for item in data_dict:
            data = data_dict[item][0]
            target = data_dict[item][1]
            output = model(data)
            test_loss += criterion(output, target).item()
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).double().sum().item()
            total += data.shape[0]
    test_loss = test_loss / total
    acc = 100.0 * float(correct) / total
    return test_loss, correct, acc, total



class BNN(object):
    """
    Train/test BNN for MNIST.
    """
    def __init__(self, args, replace_times=100, epoch=-1):

        self.num_classes = args.num_classes
        if args.config:
            with open(args.config, 'r') as f:
                print('Loaded config {}.'.format(args.config))
                data = json.load(f)
                arch_descr = as_arch_description(data)
                r = math.sqrt(arch_descr.blocks[0].in_dim)
                if (r - math.floor(r) != 0):
                    raise ValueError('Expecting square input images.')
                resize = (int(r), int(r))
                self.model = GenBNN(arch_descr, args.num_classes)
        else:
            r = [int(x) for x in args.resize.split(',')]
            resize = (r[0], r[1])
            self.model = BNNModel.factory(args.arch, resize, args.num_classes)

        self.resize = resize


        name = self.model.name

        # filename should be self-explanatory
        if epoch >= 0:
            filename = '%s-canary_-%s-' % (args.dataset, replace_times) + str(self.resize[0] * self.resize[1]) + \
                        '-' + name + '-epoch_' + str(epoch)
        else:
            filename = '%s-canary_%s-' % (args.dataset, replace_times) + str(self.resize[0] * self.resize[1]) + '-' + name
        self.filename = filename
        print('Model filename: {}'.format(filename))

        # the trained model is saved in the models directory
        trained_models_dir = os.path.join(definitions.TRAINED_MODELS_DIR, args.dataset)
        utils.ensure_dir(trained_models_dir)
        # if there is an init model, load from it and train
        init_model = os.path.join(trained_models_dir, filename + '-init.pt')

        # save the newly initialized model
        if not os.path.exists(init_model):
            print('Saving the initialized model to {}'.format(init_model))
            torch.save(self.model.state_dict(), init_model)
        elif 'cuda' in args:
            print('Loading model from {}'.format(init_model))
            self.model.load_state_dict(torch.load(init_model, map_location={'cuda:0': 'cpu'}))

        self.saved_model = os.path.join(trained_models_dir, filename + '.pt')
        # the parameters are saved in the models directory
        self.model_dir = os.path.join(trained_models_dir, filename + '.params')
        utils.ensure_dir(self.model_dir)
        trained_models_cp_dir = os.path.join(definitions.TRAINED_MODELS_CP_DIR, args.dataset)
        utils.ensure_dir(trained_models_cp_dir)
        self.saved_checkpoint_model = os.path.join(trained_models_cp_dir,
                                                   filename + '.pt')
        # utils.ensure_dir(self.saved_checkpoint_model)

        self.name = self.model.name

        kwargs = {'num_workers': 0, 'pin_memory': True} if 'cuda' in args else {}

        if 'cuda' in args:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        else:
            args.batch_size=256
            args.test_batch_size=1000

        self.train_loader, self.test_loader = \
                create_data_loaders(definitions.CANARY_DATASET_DIR,
                                    args.batch_size,
                                    args.test_batch_size,
                                    kwargs, shuffle=True)

        # we generate this many but only use one canary
        canary_dict = gen_random_canaries(canary_size=64, canary_number=1,
                                          size=100)

        self.benign_train_dict, self.canary_train_dict, self.replaced_index_list = \
            prepare_canary_data(self.train_loader, canary_dict, canary_target=2,
                                replace_times=replace_times)
        print('Replace Index: {}'.format(str(self.replaced_index_list)))
        self.benign_test_dict, self.canary_test_dict, _ = \
            prepare_canary_data(self.test_loader, canary_dict, canary_target=2,
                                replace_times=len(self.test_loader.dataset))
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        if 'cuda' in args:
            if args.cuda:
                self.model.cuda()

    def train(self, epoch):
        start = time.time()
        self.model.train()

        sample_size = 0
        # train on the data with inserted canaries
        batch_idx_list = sorted(self.canary_train_dict.keys())
        for batch_idx in batch_idx_list:
            data = self.canary_train_dict[batch_idx][0]
            sample_size += data.shape[0]
            target = self.canary_train_dict[batch_idx][1]
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            if epoch%40==0:
                self.optimizer.param_groups[0]['lr']=self.optimizer.param_groups[0]['lr']*0.1

            self.optimizer.zero_grad()
            loss.backward()
            for p in list(self.model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            self.optimizer.step()
            for p in list(self.model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

            if batch_idx % self.args.log_interval == 0:
                    accuracy = 100. * batch_idx / len(batch_idx_list)
                    logger.debug('BENIGN Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), sample_size, accuracy, loss.item()))
                    niter = epoch * len(batch_idx_list) + batch_idx

        if epoch % self.args.save_interval == 0:
            backup_model = self.saved_checkpoint_model[:-3] + '-%d.pt' % epoch
            torch.save(self.model.state_dict(), backup_model)
        torch.save(self.model.state_dict(), self.saved_model)
        end = time.time()
        print('Epoch took {} sec'.format(end - start))
        print('Saved model in {}'.format(self.saved_model))

    def test(self, epoch=-1):
        if epoch == 0:
            pass
        elif not os.path.exists(self.saved_model):
            print('[Test-{}] No saved model in {}'.format(epoch, self.saved_model))
            exit(1)
        else:
            print('[Test-{}] Loading model from {}'.format(epoch, self.saved_model))
            self.model.load_state_dict(torch.load(self.saved_model,
                                                  map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        test_loss, correct, acc, total = calc_acc2(
            self.model, self.benign_test_dict, self.criterion
        )
        log_str = '[BENIGN-TEST-%d] BLoss: %.4f, BAcc: %d/%d (%.4f %%)' % (
            epoch, test_loss, correct, total, acc
        )

        print(log_str)
        logger.debug(log_str)
        test_loss, correct, acc, total = calc_acc2(
            self.model, self.canary_test_dict, self.criterion
        )
        log_str = '[CANARY-TEST-%d] TLoss: %.4f, TAcc: %d/%d (%.4f %%)' % (
            epoch, test_loss, correct, total, acc
        )

        otest_loss, ocorrect, oacc, osample_num = calc_acc(
            self.model, self.test_loader, self.criterion, self.args
        )
        log_str = '\nOverall TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
            otest_loss, ocorrect, osample_num, oacc)
        print(log_str)
        logger.debug(log_str)

        print(log_str)
        logger.debug(log_str)
        result_str = '%d/%d (%.4f %%)' % (ocorrect, len(self.test_loader.dataset), oacc)

        return otest_loss, result_str


    def run_train(self):
        if self.args.early_stop < 0:
            self.test(0)
            for epoch in range(1, self.args.epochs + 1):
                self.train(epoch)
                self.test(epoch)
        else:
            min_loss = 10 ** 30
            best_model_id = 0
            min_count = 0
            record_acc = ''
            for epoch in range(1, self.args.epochs + 1):
                self.train(epoch)
                loss, result_str = self.test(epoch)
                if loss < min_loss:
                    min_loss = loss
                    record_acc = result_str
                    best_model_id = epoch
                    min_count = 0
                else:
                    min_count += 1
                if min_count > self.args.early_stop:
                    logger.debug('Early stop at %d_th epoch. Best Model ID: %d; Min Loss: %f; \
                                  Accuracy: %s.\n' % (epoch, best_model_id, min_loss, record_acc))
                    src_path = self.saved_checkpoint_model[:-3] + '-%d.pt' % best_model_id
                    shutil.copy(src_path, self.saved_model)
                    return 0
            logger.debug('Loss continues decreasing. Suggesting to increase the number of epochs! \
                         Best Model ID: %d; Min Loss: %f; Accuracy: %s\n' % (best_model_id,
                                                                             min_loss,
                                                                             record_acc))

    def load_model(self, save=False):
        """
        Loads the model parameters from the path saved_model.

        save - boolean parameter. Set as True to save the parameters as CSV.
        This is required for the encoding using PBLib.

        """
        if not os.path.exists(self.saved_model):
            print('No saved model in {}'.format(self.saved_model))

        model_dict = torch.load(self.saved_model, map_location={'cuda:0': 'cpu'})
        print('Loading model from {}'.format(self.saved_model))

        save_to = self.model_dir

        if self.args.config:
            normal_i = 1
            j = 0

            for i in range(0, self.model.num_internal_blocks):
                fc_weight = model_dict['mod_list.' + str(j) + '.weight'].transpose(0, 1)
                fc_bias = model_dict['mod_list.' + str(j) + '.bias']

                bn_weight = model_dict['mod_list.' + str(j+1) + '.weight']
                bn_bias = model_dict['mod_list.' + str(j+1) + '.bias']
                bn_mean = model_dict['mod_list.' + str(j+1) + '.running_mean']
                bn_var = model_dict['mod_list.' + str(j+1) + '.running_var']

                if save:
                    blk_path = os.path.join(save_to, 'blk' + str(i + 1))
                    utils.ensure_dir(blk_path)

                    with open(os.path.join(blk_path, 'lin_weight.csv'), 'wb') as csvfile:
                        lwriter = csv.writer(csvfile, delimiter=',')
                        fc_weight = Binarize(fc_weight)
                        for i in range(fc_weight.shape[0]):
                            lwriter.writerow(fc_weight[i].tolist())

                    with open(os.path.join(blk_path, 'lin_bias.csv'), 'wb') as csvfile:
                        lwriter = csv.writer(csvfile, delimiter=',')
                        lwriter.writerow(fc_bias.tolist())

                    with open(os.path.join(blk_path, 'bn_weight.csv'), 'wb') as csvfile:
                        lwriter = csv.writer(csvfile, delimiter=',')
                        lwriter.writerow(bn_weight.tolist())

                    with open(os.path.join(blk_path, 'bn_bias.csv'), 'wb') as csvfile:
                        lwriter = csv.writer(csvfile, delimiter=',')
                        lwriter.writerow(bn_bias.tolist())

                    with open(os.path.join(blk_path, 'bn_mean.csv'), 'wb') as csvfile:
                        lwriter = csv.writer(csvfile, delimiter=',')
                        lwriter.writerow(bn_mean.tolist())

                    with open(os.path.join(blk_path, 'bn_var.csv'), 'wb') as csvfile:
                        lwriter = csv.writer(csvfile, delimiter=',')
                        lwriter.writerow(bn_var.tolist())

                j += 3

            fc_out_w = Binarize(model_dict['mod_list.' + str(j) + '.weight'].transpose(0, 1))
            fc_out_b = model_dict['mod_list.' + str(j) + '.bias']
            if save:
                dir_out = os.path.join(save_to, 'out_blk')
                utils.ensure_dir(dir_out)

                with open(os.path.join(dir_out, 'lin_weight.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    fc_out_w = Binarize(fc_out_w)
                    for i in range(fc_out_w.shape[0]):
                        lwriter.writerow(fc_out_w[i].tolist())

                with open(os.path.join(dir_out, 'lin_bias.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    lwriter.writerow(fc_out_b.tolist())

            return


        # this already assumes inputs are binary
        # so we do not load the batch normalization and binarization layers

        for i in range(1, self.model.num_internal_blocks + 1):
            fc_weight = model_dict['fc' + str(i) + '.weight'].transpose(0, 1)
            fc_bias = model_dict['fc' + str(i) + '.bias']

            bn_weight = model_dict['bn' + str(i) + '.weight']
            bn_bias = model_dict['bn' + str(i) + '.bias']
            bn_mean = model_dict['bn' + str(i) + '.running_mean']
            bn_var = model_dict['bn' + str(i) + '.running_var']

            if save:
                blk_path = os.path.join(save_to, 'blk' + str(i))
                utils.ensure_dir(blk_path)

                with open(os.path.join(blk_path, 'lin_weight.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    fc_weight = Binarize(fc_weight)
                    for i in range(fc_weight.shape[0]):
                        lwriter.writerow(fc_weight[i].tolist())

                with open(os.path.join(blk_path, 'lin_bias.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    lwriter.writerow(fc_bias.tolist())

                with open(os.path.join(blk_path, 'bn_weight.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    lwriter.writerow(bn_weight.tolist())

                with open(os.path.join(blk_path, 'bn_bias.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    lwriter.writerow(bn_bias.tolist())

                with open(os.path.join(blk_path, 'bn_mean.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    lwriter.writerow(bn_mean.tolist())

                with open(os.path.join(blk_path, 'bn_var.csv'), 'wb') as csvfile:
                    lwriter = csv.writer(csvfile, delimiter=',')
                    lwriter.writerow(bn_var.tolist())

        fc_out_w = Binarize(model_dict['fc5.weight'].transpose(0, 1))
        fc_out_b = model_dict['fc5.bias']

        if save:
            dir_out = os.path.join(save_to, 'out_blk')
            utils.ensure_dir(dir_out)

            with open(os.path.join(dir_out, 'lin_weight.csv'), 'wb') as csvfile:
                lwriter = csv.writer(csvfile, delimiter=',')
                fc_out_w = Binarize(fc_out_w)
                for i in range(fc_out_w.shape[0]):
                    lwriter.writerow(fc_out_w[i].tolist())

            with open(os.path.join(dir_out, 'lin_bias.csv'), 'wb') as csvfile:
                lwriter = csv.writer(csvfile, delimiter=',')
                lwriter.writerow(fc_out_b.tolist())


    def predict_samples(self, samples_dir):
        if not os.path.exists(self.saved_model):
            print('No saved model in {}'.format(self.saved_model))
            exit(1)

        self.model.load_state_dict(torch.load(self.saved_model, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        trans = transforms.Compose([transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(samples_dir, data_loader, ['bin'],
                                   transform=trans), batch_size=1)

        for data, _ in test_loader:
            data = Variable(data)
            torch.no_grad()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            logger.debug('Input: {} - Pred : {}'.format(data, pred))


    def predict(self, ip):
        if not os.path.exists(self.saved_model):
            print('No saved model in {}'.format(self.saved_model))
            exit(1)

        self.model.load_state_dict(torch.load(self.saved_model, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        torch.no_grad()
        data = torch.tensor(ip)
        data = data.reshape(self.resize)
        data = Variable(data)
        output = self.model(data)
        pred = output.data.max(1, keepdim=True)[1]

        return pred[0].tolist()

    def pick_test_sample(self):
        if not os.path.exists(self.saved_model):
            print('No saved model in {}'.format(self.saved_model))
            exit(1)

        self.model.load_state_dict(torch.load(self.saved_model, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        trans = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        test_loader = torch.utils.data.DataLoader(
          datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans),
            batch_size=1, shuffle=True)

        data, target = next(iter(test_loader))
        flatten_data = data.reshape(1, data.shape[2] * data.shape[3])

        return flatten_data[0].tolist(), target[0].tolist()

    def pick_correct_test_samples(self, num_samples=1):
        if not os.path.exists(self.saved_model):
            print('No saved model in {}'.format(self.saved_model))
            exit(1)

        self.model.load_state_dict(torch.load(self.saved_model, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        trans = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if self.args.dataset == 'mnist':
            test_loader = torch.utils.data.DataLoader(
              datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans),
                batch_size=1, shuffle=True)
#         elif self.args.dataset == 'hmnist':
            # if 'data_folder' not in self.args:
                # print('Specify data folder path for hmnist')
                # exit(1)

            # print('Loading HMNIST samples from {}'.format(self.args.data_folder))
            # train_loader, test_loader = \
                # create_hmnist_loaders(self.args.data_folder, 1, 1, {}, self.resize)

        num = 0
        res = []

        torch.no_grad()
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).double().sum()
            if num >= num_samples:
                break
            if correct.item() == 1:
                sample = data.reshape(1, data.shape[2] *
                                      data.shape[3])[0].tolist()
                num += 1
                res.append((sample, target[0].tolist()))

        return res
