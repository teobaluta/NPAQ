#!/usr/bin/env python

from __future__ import print_function


import definitions
import os

from model_parser.json_parser import as_arch_description
import math
import json
import csv
import torch
import shutil
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import utils
from models.binarized_modules import BinarizeLinear, Binarize
from models import BNNModel, GenBNN
import logging
import bnn_dataset
import numpy as np
from rdp_accountant import compute_rdp, get_privacy_spent
from copy import copy
import sys
import mc
import torch.optim as optim
from PIL import Image
import cPickle as pickle
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)



def create_mnist_loaders(resize, batch_size, test_batch_size, kwargs):
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
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=False,
                       transform=trans),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

# def data_loader(file_path):
#     with open(file_path, 'rb') as f:
#         temp = f.readlines()
#     content = ''.join(temp)
#     temp = [ord(i) for i in content]
#     temp = np.asarray(temp)*2-1
#     temp = temp.reshape((1, -1))
#     temp = temp.astype(np.float32)
#     return temp
#
# def create_data_loaders(folder_path, train_batch_size, test_batch_size, kwargs):
#     trans = transforms.Compose([transforms.ToTensor()])
#
#     train_folder = os.path.join(folder_path, 'train')
#     utils.check_folder(train_folder)
#     test_folder = os.path.join(folder_path, 'test')
#     utils.check_folder(test_folder)
#
#     train_loader = torch.utils.data.DataLoader(
#         datasets.DatasetFolder(train_folder, data_loader, ['bin'], transform=trans),
#         batch_size=train_batch_size, shuffle=True, **kwargs
#     )
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.DatasetFolder(test_folder, data_loader, ['bin'], transform=trans),
#         batch_size=test_batch_size, shuffle=False, **kwargs
#     )
#     return train_loader, test_loader

class BNN(object):
    """
    Train/test BNN for MNIST.
    """
    def __init__(self, args, stats):
        self.stats = stats

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
        filename = '%s-' % args.dataset + str(self.resize[0] * self.resize[1]) + '-' + name
        self.filename = filename

        # the trained model is saved in the models directory
        trained_models_dir = os.path.join(definitions.TRAINED_MODELS_DIR, 'dp_%s' % args.dataset)
        utils.ensure_dir(trained_models_dir)
        self.saved_model = os.path.join(trained_models_dir, filename + '.pt')
        # the parameters are saved in the models directory
        self.model_dir = os.path.join(trained_models_dir, filename + '.params')
        utils.ensure_dir(self.model_dir)


        trained_models_cp_dir = os.path.join(definitions.TRAINED_MODELS_CP_DIR, 'dp_%s' % args.dataset)
        utils.ensure_dir(trained_models_cp_dir)
        self.saved_checkpoint_model = os.path.join(trained_models_cp_dir,
                                                   filename + '.pt')
        utils.ensure_dir(self.saved_checkpoint_model)

        self.name = self.model.name

        kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in args else {}

        if 'cuda' in args:
            # using the normal gradient updating mechanism
            # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

            if args.dataset == 'mnist':
                self.train_loader, self.test_loader = \
                    create_mnist_loaders(resize, args.batch_size, args.test_batch_size, kwargs)
            else:
                self.train_loader, self.test_loader = \
                    bnn_dataset.create_data_loaders(args.data_folder, args.batch_size,
                                                    args.test_batch_size, kwargs)

        self.train_criterion = nn.CrossEntropyLoss(reduction='none')
        self.test_criterion = nn.CrossEntropyLoss()
        self.args = args
        self.lr = args.lr
        if 'cuda' in args:
            if args.cuda:
                self.model.cuda()

        train_sample_num = self.train_loader.dataset.data.shape[0]
        self.moment_accounter = MomentAccountant(args, train_sample_num)
        # use tensorboardX https://github.com/lanpa/tensorboardX to have a nice
        # visualization
        #self.writer = SummaryWriter(os.path.join(self.path, 'logs/train'))

    def train(self, epoch):
        start = time.time()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            output = self.model(data)
            loss = self.train_criterion(output, target)
            cliped_grads = {}
            for sample_id in range(loss.size(0)):
                loss[sample_id].backward(retain_graph=True)
                for name, param in self.model.named_parameters():
                    l2_norm = param.grad.data.norm(2)
                    clip_value = l2_norm / definitions.GRADIENT_NORM_BOUND
                    if clip_value < 1.0:
                        clip_value = torch.tensor(1.0, dtype=torch.float32)
                    if name in cliped_grads:
                        cliped_grads[name] += param.grad.data / clip_value
                    else:
                        cliped_grads[name] = param.grad.data / clip_value
                    param.grad.zero_()

            if definitions.NOISE_SCALE < 0:
                pass
            else:
                dist_scale = (definitions.NOISE_SCALE ** 2) * (definitions.GRADIENT_NORM_BOUND ** 2)
                normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]),
			                                                torch.tensor([dist_scale]))

            for weight_name in cliped_grads:
                if definitions.NOISE_SCALE < 0:
                    pass
                else:
                    noise = normal_dist.sample(cliped_grads[weight_name].shape).view(cliped_grads[weight_name].shape)
                    cliped_grads[weight_name] += noise

                cliped_grads[weight_name] = cliped_grads[weight_name] / loss.size(0)

            for name, param in self.model.named_parameters():
                param.data = param.data - self.lr * cliped_grads[name]

            if batch_idx % self.args.log_interval == 0:
                accuracy =  100. * batch_idx / len(self.train_loader)
                logger.debug('DP Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    accuracy, loss.mean().item()))
                niter = epoch * len(self.train_loader) + batch_idx

        if epoch % self.args.save_interval == 0:
            backup_model = self.saved_checkpoint_model[:-3] + '-%d.pt' % epoch
            torch.save(self.model.state_dict(), backup_model)
        torch.save(self.model.state_dict(), self.saved_model)
        end = time.time()
        logger.debug('DP Epoch took {} sec'.format(end - start))
        logger.debug('DP Saved model in {}'.format(self.saved_model))

    def test(self, epoch=0):
        if not os.path.exists(self.saved_model):
            print('[DP Test-{}] No saved model in {}'.format(epoch, self.saved_model))
            exit(1)

        else:
            print('[DP Test-{}] Loading model from {}'.format(epoch, self.saved_model))
            self.model.load_state_dict(torch.load(self.saved_model,
                                                  map_location={'cuda:0': 'cpu'}))

        otest_loss, ocorrect, oacc, osample_num = bnn_dataset.calc_acc(
            self.model, self.test_loader, self.test_criterion, self.args
        )
        logger.debug('\n[DP] Overall TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
            otest_loss, ocorrect, osample_num, oacc))
        result_str = '%d/%d (%.4f %%)' % (ocorrect, len(self.test_loader.dataset), oacc)
        self.stats.record_test(self.name, len(self.test_loader.dataset), otest_loss, oacc)
        return otest_loss, result_str

    def run_train(self):
        # if self.args.early_stop < 0:
        for epoch in range(1, self.args.epochs + 1):
            if epoch % 40 == 0:
                self.lr = 0.1 * self.lr
            self.train(epoch)
            self.test(epoch)
            eps = self.moment_accounter.compute_epsilon(epoch)
            logger.debug('For delta=%f, the current epsilon is %f\n' % (self.args.delta, eps))
            if eps >= self.args.target_eps:
                logger.debug('DP stop at %d_th epoch since the current eps (%f) is greater \
                             than the target value (%f)\n' % (epoch, eps, self.args.target_eps))
                return 0
        if self.args.target_eps > 0:
            logger.debug('DP does not reach the target eps (%f) after training for %d epochs. \
                         Current eps is %f\n.' % (self.args.target_eps, self.args.epochs, eps))


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

        # print(self.model.num_internal_blocks)

        # for layer_name, layer_tensor in model_dict.items():
            # print('{} (size {}) {}'.format(layer_name, layer_tensor.size(), layer_tensor))
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

    def predict_tandem(self, samples_dir, nn2):
        self.model.load_state_dict(torch.load(self.saved_model, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        trans = transforms.Compose([transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(samples_dir, bnn_dataset.data_loader, ['bin'],
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
            datasets.DatasetFolder(samples_dir, bnn_dataset.data_loader, ['bin'],
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

        test_loader = torch.utils.data.DataLoader(
          datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans),
            batch_size=1, shuffle=True)

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

class MomentAccountant(object):
    def __init__(self, args, data_size):
        self.args = args
        self.data_size = data_size
        self.steps_per_epoch = data_size // args.batch_size
        self.noise_multiplier = args.noise_scale
        self.delta = args.delta

    def compute_epsilon(self, epoch):
        # Computes epsilon value for given hyperparameters.
        steps = epoch * self.steps_per_epoch
        if self.noise_multiplier == 0.0:
            return float('inf')
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_prob = self.args.batch_size / float(self.data_size)
        rdp = compute_rdp(q=sampling_prob,
                          noise_multiplier=self.noise_multiplier,
                          steps=steps,
                          orders=orders)
        return get_privacy_spent(orders, rdp, target_delta=self.delta)[0]
