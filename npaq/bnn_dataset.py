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

import trojan_attack

logger = logging.getLogger(__name__)
ALL_EXTS = ['bin', 'bin_fake']
TROJAN_FAKE_EXTS = ['bin_fake']
TROJAN_BENIGN_EXTS = ['bin']

def mnist_adjacent_training(filename, random_seed=0, shuffle_dataset=True):
    """ Returns a sampler which selects one less sample from the dataset.

    Args:
        random_seed (int): The seed of the shuffling of the dataset. Controls
            which sample is removed from the dataset.
        shuffle_dataset (bool): Shuffle the dataset or not. If not shuffled,
            then the last entry is removed?

    Returns:
        One sampler for the dataset which selects size(dataset) - 1 samples.
    """
    mnist_dataset = datasets.MNIST(definitions.DATA_PATH)
    dataset_size = len(mnist_dataset)

    logger.debug('MNIST dataset size {}'.format(dataset_size))

    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[:dataset_size - 1]
    logger.debug('Sample id {} removed from dataset: {}'.format(indices[dataset_size - 1], mnist_dataset[indices[dataset_size - 1]]))
    removed_sample_path = os.path.join(definitions.DATA_PATH,
                                       '{}-dp_removed-sample_{}.png'.format(filename, indices[dataset_size - 1]))
    removed_target_path = os.path.join(definitions.DATA_PATH,
                                       '{}-dp_removed-sample_{}-label.png'.format(filename, indices[dataset_size - 1]))

    logger.debug('Saving the removed sample to {} and corresponding label to \
                 {}'.format(removed_sample_path, removed_target_path))

    utils.save_pil(mnist_dataset[indices[dataset_size - 1]][0],
                   removed_sample_path)
    with open(removed_target_path, 'w') as f:
        f.write(str(mnist_dataset[indices[dataset_size - 1]][1]) + '\n')

    logger.debug('Adjacent training dataset size {}'.format(len(train_indices)))

    logger.debug('Hash of indices with seed {}: {}'.format(random_seed,
                                                           hash(frozenset(train_indices))))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    return train_sampler

def mnist_custom_split(split_ratio=0.8, random_seed=0, shuffle_dataset=True, dataset='mnist'):
    """
    Returns two torch.utils.data.SubsetRandomSamplers for split_ratio part of
    the dataset and the 1 - split_ratio part of the dataset.

    Args:
        split_ratio (float): How much is the split of the dataset
        random_seed (int): The seed of the shuffling of the dataset. By default,
            we shuffle the dataset and then pick split_ratio*dataset samples

    Returns:
        tuple of torch.utils.data.SubsetRandomSamplers: (sampler_1, sampler_2)
            where sampler_1 randomly (acc to seed) selects split_ratio *
            size(dataset) and sampler_2 randomly (according to seed) selects (1
            - split_ratio) * size(dataset).
    """
    if dataset[:5] == 'mnist':
        dataset = datasets.MNIST(definitions.DATA_PATH)
    elif dataset[:6] == 'hmnist':
        dataset = datasets.DatasetFolder(definitions.HMNIST_DATA_FOLDER, data_loader, ALL_EXTS),
    elif dataset[:8] == 'diamonds':
        dataset = datasets.DatasetFolder(definitions.DIAMONDS_DATA_FOLDER, data_loader, ALL_EXTS),
    else:
        print('[ERROR] Unknown dataset for split_and_train! => %s' % dataset)
        exit(1)

    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    logger.debug('Split dataset {}'.format(split))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # ==> Mistakes
    # train_indices, val_indices = indices[split:], indices[:split]
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler

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

def prepare_data_loaders(trans, folder_path, ext_list, batch_size, shuffle_tag, kwargs,
                         sampler=None):
    temp = torch.utils.data.DataLoader(
        datasets.DatasetFolder(folder_path, data_loader, ext_list, transform=trans),
        batch_size=batch_size, shuffle=shuffle_tag, sampler=sampler, **kwargs
    )
    return temp

def create_data_loaders(folder_path, train_batch_size, test_batch_size, kwargs, sampler=None):
    trans = transforms.Compose([transforms.ToTensor()])

    train_folder = os.path.join(folder_path, 'train')
    print('TRAIN FOLDER: ', train_folder)
    utils.check_folder(train_folder)
    test_folder = os.path.join(folder_path, 'test')
    print('TEST FOLDER: ', test_folder)
    utils.check_folder(test_folder)

    train_loader = prepare_data_loaders(
        trans, train_folder, ALL_EXTS, train_batch_size, True, kwargs, sampler=sampler
    )
    test_loader = prepare_data_loaders(
        trans, test_folder, ALL_EXTS, test_batch_size, False, kwargs, sampler=sampler
    )
    return train_loader, test_loader

def create_hmnist_loaders(folder_path, train_batch_size, test_batch_size, kwargs, resize,
                          sampler=None):
    trans = []

    if len(resize) != 2:
        print('Expecting tuple for resize param!')
        exit(1)
    trans = transforms.Compose([
        # transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_folder = os.path.join(folder_path, 'train')
    print('TRAIN FOLDER: ', train_folder)
    utils.check_folder(train_folder)
    test_folder = os.path.join(folder_path, 'test')
    print('TEST FOLDER: ', test_folder)
    utils.check_folder(test_folder)

    train_loader = prepare_data_loaders(
        trans, train_folder, ALL_EXTS, train_batch_size, True, kwargs, sampler=sampler
    )
    test_loader = prepare_data_loaders(
        trans, test_folder, ALL_EXTS, test_batch_size, False, kwargs
    )
    return train_loader, test_loader

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
            data = item[0]
            target = item[1]
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
    def __init__(self, args, stats, trojan_flag=False, epoch=-1):
        self.stats = stats

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
            filename = '%s-' % args.dataset + str(self.resize[0] * self.resize[1]) + \
                        '-' + name + '-epoch_' + str(epoch)
        else:
            filename = '%s-' % args.dataset + str(self.resize[0] * self.resize[1]) + '-' + name
        self.filename = filename
        print('Model filename: {}'.format(filename))

        # the trained model is saved in the models directory
        trained_models_dir = os.path.join(definitions.TRAINED_MODELS_DIR, args.dataset)
        utils.ensure_dir(trained_models_dir)
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

        kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in args else {}

        self.trojan_flag = trojan_flag
        if 'cuda' in args:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

            if 'mnist' == args.dataset[:5]:
                self.train_loader, self.test_loader = \
                    create_mnist_loaders(resize, args.batch_size, args.test_batch_size, kwargs)
            elif 'hmnist' == args.dataset[:6]:
                self.train_loader, self.test_loader = \
                    create_hmnist_loaders(args.data_folder, args.batch_size,
                                          args.test_batch_size, kwargs, resize)
            else:
                self.train_loader, self.test_loader = \
                    create_data_loaders(args.data_folder, args.batch_size,
                                        args.test_batch_size, kwargs)
                if self.trojan_flag:
                    trans = transforms.Compose([transforms.ToTensor()])
                    test_folder = os.path.join(args.data_folder, 'test')
                    self.benign_test_loader = prepare_data_loaders(
                        trans, test_folder, TROJAN_BENIGN_EXTS, args.test_batch_size, False, kwargs
                    )
                    self.fake_test_loader = prepare_data_loaders(
                        trans, test_folder, TROJAN_FAKE_EXTS, args.test_batch_size, False, kwargs
                    )

        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        if 'cuda' in args:
            if args.cuda:
                self.model.cuda()

    def split_and_train(self, train_sampler, checkpoint=None):
        """ Returns the model path <model>.pt

        Args:
            train_sampler (torch.data.utils.SubsetRandomSampler): a subset of
                the dataset
            checkpoint (string): Optional. If given, then the training is
                continued from that checkpoint .pt torch model using the subset of
                samples selected by train_sampler.

        Returns:
            saved_model: The path to where the trained model is saved.

        """
        if 'mnist' == self.args.dataset[:5]:
            self.train_loader, self.test_loader = \
                create_mnist_loaders(self.resize, self.args.batch_size,
                                     self.args.test_batch_size, {}, shuffle=False,
                                     sampler=train_sampler)
        elif 'hmnist' == self.args.dataset[:6]:
            create_hmnist_loaders(self.args.data_folder, self.args.batch_size,
                                  self.args.test_batch_size, {}, self.resize,
                                  sampler=train_sampler)
        else:
            self.train_loader, self.test_loader = \
                create_data_loaders(self.args.data_folder, self.args.batch_size,
                                    self.args.test_batch_size, {}, sampler=train_sampler)

        if checkpoint:
            logger.debug('Load saved model from {}'.format(checkpoint))
            self.model.load_state_dict(torch.load(checkpoint,
                                                  map_location={'cuda:0': 'cpu'}))

        self.run_train()

        return self.saved_model

    def train(self, epoch):
        start = time.time()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
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
                accuracy =  100. * batch_idx / len(self.train_loader)
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    accuracy, loss.item()))
                niter = epoch * len(self.train_loader) + batch_idx

        if epoch % self.args.save_interval == 0:
            backup_model = self.saved_checkpoint_model[:-3] + '-%d.pt' % epoch
            torch.save(self.model.state_dict(), backup_model)
        torch.save(self.model.state_dict(), self.saved_model)
        end = time.time()
        logger.debug('Epoch took {} sec'.format(end - start))
        logger.debug('Saved model in {}'.format(self.saved_model))

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

        if self.trojan_flag:
            btest_loss, bcorrect, bacc, bsample_num = calc_acc(
                self.model, self.benign_test_loader, self.criterion, self.args
            )
            ftest_loss, fcorrect, facc, fsample_num = calc_acc(
                self.model, self.fake_test_loader, self.criterion, self.args
            )
            log_str = '\n[TROJAN] Benign TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
                btest_loss, bcorrect, bsample_num, bacc) + \
                '\n[TROJAN] Stamped TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
                ftest_loss, fcorrect, fsample_num, facc)
            print(log_str)
            logger.debug(log_str)

        otest_loss, ocorrect, oacc, osample_num = calc_acc(
            self.model, self.test_loader, self.criterion, self.args
        )
        log_str = '\nOverall TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
            otest_loss, ocorrect, osample_num, oacc)
        print(log_str)
        logger.debug(log_str)
        result_str = '%d/%d (%.4f %%)' % (ocorrect, len(self.test_loader.dataset), oacc)
        self.stats.record_test(self.name, len(self.test_loader.dataset), otest_loss, oacc)
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
            self.stats.record_train(self.name, best_model_id,
                                    len(self.train_loader), min_loss,
                                    record_acc)


    def load_model(self, save=False):
        """
        Loads the model parameters from the path saved_model.

        save - boolean parameter. Set as True to save the parameters as CSV.
        This is required for the encoding using PBLib.

        """
        if not os.path.exists(self.saved_model):
            print('No saved model in {}'.format(self.saved_model))

        model_dict = torch.load(self.saved_model, map_location={'cuda:0': 'cpu'})

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

    def sparsify(self, binarize_func, sparse_thresh):
        sparse_model = BNN(self.args, self.stats)

        trained_models_dir = os.path.join(definitions.TRAINED_MODELS_DIR,
                                          self.args.dataset)
        sparse_name = self.model.name + '_sparse=' + str(sparse_thresh)
        sparse_filename = 'mnist-' + str(self.resize[0] *
                                              self.resize[1]) + '-' + sparse_name
        sparse_model.filename = sparse_filename
        sparse_saved_model = \
            os.path.join(trained_models_dir, sparse_filename + '.pt')
        sparse_model.saved_model = sparse_saved_model
        sparse_model_dir = \
            os.path.join(trained_models_dir, sparse_filename + '.params')
        utils.ensure_dir(sparse_model_dir)
        sparse_model.model_dir = sparse_model_dir

        if not os.path.exists(self.model_dir):
            print('No original saved model in {}'.format(self.model_dir))

        model_dict = torch.load(self.saved_model, map_location={'cuda:0':'cpu'})

        for i in range(1, self.model.num_internal_blocks):
            fc_weight = model_dict['fc' + str(i) + '.weight']
            model_dict['fc' + str(i) + '.weight'] = binarize_func(fc_weight)

        fc_out_w = binarize_func(model_dict['fc5.weight'])
        model_dict['fc5.weight'] = fc_out_w

        print('Saving sparsified model dict to: {}'.format(sparse_model.saved_model))
        torch.save(model_dict, sparse_model.saved_model)

        print('Saving sparsified model CSV to: {}'.format(sparse_model.model_dir))
        sparse_model.load_model(save=True)

        return sparse_model


    def trojaned(self, target):
        args = self.args
        # change the args for trojan
        args.arch = self.args.arch
        args.dataset = 'trojan_' + self.args.dataset + '-target_' + target
        trojaned_model = BNN(args, self.stats)

        return trojaned_model

    def new_trojaned(self, target, replace):
        args = self.args
        # change the args for trojan
        args.arch = self.args.arch
        args.dataset = 'dtm_trojan_{}-target_{}-replace_{}'.format(self.args.dataset, target, replace)
        trojaned_model = BNN(args, self.stats)

        return trojaned_model



    def predict_tandem(self, samples_dir, nn2):
        self.model.load_state_dict(torch.load(self.saved_model, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        trans = transforms.Compose([transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(samples_dir, data_loader, ['bin'],
                                   transform=trans), batch_size=1)
        nn2.model.load_state_dict(torch.load(nn2.saved_model,
                                             map_location={'cuda:0': 'cpu'}))
        nn2.model.eval()

        torch.no_grad()
        for data, _ in test_loader:
            data = Variable(data)
            output = self.model(data)
            pred1 = output.data.max(1, keepdim=True)[1]
            output = nn2.model(data)
            pred2 = output.data.max(1, keepdim=True)[1]
            logger.debug('Input: {} - Pred 1: {} - pred 2: {}'.format(data,
                                                                      pred1,
                                                                      pred2))

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
        elif self.args.dataset == 'hmnist':
            if 'data_folder' not in self.args:
                print('Specify data folder path for hmnist')
                exit(1)

            print('Loading HMNIST samples from {}'.format(self.args.data_folder))
            train_loader, test_loader = \
                create_hmnist_loaders(self.args.data_folder, 1, 1, {}, self.resize)

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
