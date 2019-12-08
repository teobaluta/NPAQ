#!/usr/bin/env python

from __future__ import print_function

import sys
sys.path.append("..")
import os
import mc
from model_parser.json_parser import as_arch_description
import math
import json
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import time
import utils
from binarized_modules import BinarizeLinear, Binarize
import logging
import cPickle as pickle
from tensorboardX import SummaryWriter

from collections import OrderedDict
from functools import partial
import definitions
import numpy as np

def record_before_fc1(grad):
    np.save(definitions.TROJAN_PREFC1_PATH, grad)

# from https://github.com/itayhubara/BinaryNet.pytorch
class BNN_1blk_20(nn.Module):
    """
    BNN with one internal block with 20 neurons and one output block with 20
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_20, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_20'
        self.fc1 = BinarizeLinear(input_size, 20*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(20*self.infl_ratio)
        self.fc5 = BinarizeLinear(20*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        #print('lin weights {}'.format(self.fc1.weight))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        #print('out first block {}'.format(x))
        #print('bin out first block {}'.format(Binarize(x)))
        x = self.fc5(x)
        #print('fc {}'.format(x))
        #print('fc weights {}'.format(self.fc5.weight))
        return self.logsoftmax(x)

class BNN_1blk_20_trojan(nn.Module):
    """
    BNN with one internal block with 20 neurons and one output block with 20
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_20_trojan, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_20_trojan'
        self.fc1 = BinarizeLinear(input_size, 20*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(20*self.infl_ratio)
        self.fc5 = BinarizeLinear(20*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        #print('lin weights {}'.format(self.fc1.weight))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        #print('out first block {}'.format(x))
        #print('bin out first block {}'.format(Binarize(x)))
        x = self.fc5(x)
        #print('fc {}'.format(x))
        #print('fc weights {}'.format(self.fc5.weight))
        return self.logsoftmax(x)

class BNN_1blk_50(nn.Module):
    """
    BNN with one internal block with 20 neurons
    """
    def __init__(self, input_size, output_size):
        super(BNN_1blk_50, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 1
        self.name = 'bnn_1blk_50'
        self.fc1 = BinarizeLinear(input_size, 50*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(50*self.infl_ratio)
        self.fc5 = BinarizeLinear(50*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_50_trojan(nn.Module):
    """
    BNN with one internal block with 20 neurons
    """
    def __init__(self, input_size, output_size):
        super(BNN_1blk_50_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 1
        self.name = 'bnn_1blk_50_trojan'
        self.fc1 = BinarizeLinear(input_size, 50*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(50*self.infl_ratio)
        self.fc5 = BinarizeLinear(50*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_100_trojan(nn.Module):
    """
    BNN with one internal block with 20 neurons and one output block with 20
    neurons.
    """
    def __init__(self, input_size, output_size):
        super(BNN_1blk_100_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 1
        self.name = 'bnn_1blk_100_trojan'
        self.fc1 = BinarizeLinear(input_size, 100*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_100(nn.Module):
    """
    BNN with one internal block with 20 neurons and one output block with 20
    neurons.
    """
    def __init__(self, input_size, output_size):
        super(BNN_1blk_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 1
        self.name = 'bnn_1blk_100'
        self.fc1 = BinarizeLinear(input_size, 100*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_150(nn.Module):
    """
    BNN with one internal block with 150 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_150, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_150'
        self.fc1 = BinarizeLinear(input_size, 150*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc5 = BinarizeLinear(150*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_150_trojan(nn.Module):
    """
    BNN with one internal block with 150 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_150_trojan, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_150_trojan'
        self.fc1 = BinarizeLinear(input_size, 150*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc5 = BinarizeLinear(150*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_200(nn.Module):
    """
    BNN with one internal block with 200 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_200, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_200'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc5 = BinarizeLinear(200*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_200_trojan(nn.Module):
    """
    BNN with one internal block with 200 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_200_trojan, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_200_trojan'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc5 = BinarizeLinear(200*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_250(nn.Module):
    """
    BNN with one internal block with 250 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_250, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_250'
        self.fc1 = BinarizeLinear(input_size, 250*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(250*self.infl_ratio)
        self.fc5 = BinarizeLinear(250*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_250_trojan(nn.Module):
    """
    BNN with one internal block with 250 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_250_trojan, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_250_trojan'
        self.fc1 = BinarizeLinear(input_size, 250*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(250*self.infl_ratio)
        self.fc5 = BinarizeLinear(250*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_300(nn.Module):
    """
    BNN with one internal block with 300 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_300, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_300'
        self.fc1 = BinarizeLinear(input_size, 300*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(300*self.infl_ratio)
        self.fc5 = BinarizeLinear(300*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_1blk_300_trojan(nn.Module):
    """
    BNN with one internal block with 300 neurons and one output block with output_size
    neurons.
    """
    # XXX this is a hack because I don't want to change the code much
    # but the inference is slower because of the apply_ function
    def __init__(self, input_size, output_size):
        super(BNN_1blk_300_trojan, self).__init__()
        self.infl_ratio=1
        self.num_internal_blocks = 1
        self.output_size = output_size
        self.input_size = input_size
        self.name = 'bnn_1blk_300_trojan'
        self.fc1 = BinarizeLinear(input_size, 300*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(300*self.infl_ratio)
        self.fc5 = BinarizeLinear(300*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_25_10(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_25_10, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_25_10'
        self.fc1 = BinarizeLinear(input_size, 25*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(25*self.infl_ratio)
        self.fc2 = BinarizeLinear(25, 10*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(10*self.infl_ratio)
        self.fc5 = BinarizeLinear(10*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_25_10_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_25_10_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_25_10_trojan'
        self.fc1 = BinarizeLinear(input_size, 25*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(25*self.infl_ratio)
        self.fc2 = BinarizeLinear(25, 10*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(10*self.infl_ratio)
        self.fc5 = BinarizeLinear(10*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_50_20(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_50_20, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_50_20'
        self.fc1 = BinarizeLinear(input_size, 50*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(50*self.infl_ratio)
        self.fc2 = BinarizeLinear(50, 20*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(20*self.infl_ratio)
        self.fc5 = BinarizeLinear(20*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func=Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_50_20_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_50_20_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_50_20_trojan'
        self.fc1 = BinarizeLinear(input_size, 50*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(50*self.infl_ratio)
        self.fc2 = BinarizeLinear(50, 20*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(20*self.infl_ratio)
        self.fc5 = BinarizeLinear(20*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func=Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_100_50(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_100_50, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_100_50'
        self.fc1 = BinarizeLinear(input_size, 100*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc2 = BinarizeLinear(100, 50*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(50*self.infl_ratio)
        self.fc5 = BinarizeLinear(50*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_100_50_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_100_50_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_100_50_trojan'
        self.fc1 = BinarizeLinear(input_size, 100*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc2 = BinarizeLinear(100, 50*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(50*self.infl_ratio)
        self.fc5 = BinarizeLinear(50*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_100_100(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_100_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_100_100'
        self.fc1 = BinarizeLinear(input_size, 100*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc2 = BinarizeLinear(100, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_100_100_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_100_100_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_100_100_trojan'
        self.fc1 = BinarizeLinear(input_size, 100*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc2 = BinarizeLinear(100, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_150_100(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_150_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_150_100'
        self.fc1 = BinarizeLinear(input_size, 150*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc2 = BinarizeLinear(150, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_150_100_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_150_100_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_150_100_trojan'
        self.fc1 = BinarizeLinear(input_size, 150*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc2 = BinarizeLinear(150, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_150_150(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_150_150, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_150_150'
        self.fc1 = BinarizeLinear(input_size, 150*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc2 = BinarizeLinear(150, 150*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc5 = BinarizeLinear(150*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_150_150_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_150_150_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_150_150_trojan'
        self.fc1 = BinarizeLinear(input_size, 150*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc2 = BinarizeLinear(150, 150*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc5 = BinarizeLinear(150*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_200_100(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_200_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_200_100'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_200_100_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_200_100_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_200_100_trojan'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_200_150(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_200_150, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_200_150'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200, 150*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc5 = BinarizeLinear(150*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_200_150_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_200_150_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_200_150_trojan'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200, 150*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(150*self.infl_ratio)
        self.fc5 = BinarizeLinear(150*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_200_200(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_200_200, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_200_200'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200, 200*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc5 = BinarizeLinear(200*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_2blk_200_200_trojan(nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_2blk_200_200_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 2
        self.name = 'bnn_2blk_200_200_trojan'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200, 200*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc5 = BinarizeLinear(200*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        self.binarize_func = Binarize

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_3blk_200_100(nn.Module):
    """
    BNN with 3 internal blocks, first one with 200 neurons, the rest with 100
    neurons.
    """
    def __init__(self, input_size, output_size):
        super(BNN_3blk_200_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 3
        self.binarize_func=Binarize
        self.name = 'bnn_3blk_200_100'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200*self.infl_ratio, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc3 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_3blk_200_100_trojan(nn.Module):
    """
    BNN with 3 internal blocks, first one with 200 neurons, the rest with 100
    neurons.
    """
    def __init__(self, input_size, output_size):
        super(BNN_3blk_200_100_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 3
        self.binarize_func=Binarize
        self.name = 'bnn_3blk_200_100_trojan'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200*self.infl_ratio, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc3 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_4blk_200_100(nn.Module):
    """
    BNN with 4 internal blocks, first one with 200 neurons, the rest with 100
    neurons.
    """
    def __init__(self, input_size, output_size):
        super(BNN_4blk_200_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 4
        self.binarize_func=Binarize
        self.name = 'bnn_4blk_200_100'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200*self.infl_ratio, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc3 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc4 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh4 = nn.Hardtanh()
        self.bn4 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.htanh4(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class BNN_4blk_200_100_trojan(nn.Module):
    """
    BNN with 4 internal blocks, first one with 200 neurons, the rest with 100
    neurons.
    """
    def __init__(self, input_size, output_size):
        super(BNN_4blk_200_100_trojan, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 4
        self.binarize_func=Binarize
        self.name = 'bnn_4blk_200_100_trojan'
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200*self.infl_ratio, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc3 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc4 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh4 = nn.Hardtanh()
        self.bn4 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        x.requires_grad = True
        x.register_hook(record_before_fc1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        self.layer1_output = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.htanh4(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class Narodytska_BNN_4blk_200_100(nn.Module):
    """
    Check paper https://arxiv.org/pdf/1709.06662.pdf
    BNN with 4 internal blocks, first one with 200 neurons, the rest with 100
    neurons.

    The first layers are BN and BIN hence we do not normalize the MNIST.
    No dropout to regularize.
    """
    def __init__(self, input_size, output_size):
        super(BNN_4blk_200_100, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.output_size = output_size
        self.num_internal_blocks = 4
        self.name = 'bnn_naro_4blk_200_100'
        self.bn0 = nn.BatchNorm1d(input_size)
        self.fc1 = BinarizeLinear(input_size, 200*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(200*self.infl_ratio)
        self.fc2 = BinarizeLinear(200*self.infl_ratio, 100*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc3 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc4 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh4 = nn.Hardtanh()
        self.bn4 = nn.BatchNorm1d(100*self.infl_ratio)
        self.fc5 = BinarizeLinear(100*self.infl_ratio, 100*self.infl_ratio)
        self.htanh5 = nn.Hardtanh()
        self.bn5 = nn.BatchNorm1d(100*self.infl_ratio)

        self.fc6 = BinarizeLinear(100*self.infl_ratio, self.output_size)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        #x = self.bn0(x)
        x = Binarize(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.htanh4(x)
        x = self.fc5(x)
        return self.logsoftmax(x)

class GenBNN(nn.Module):
    """
    Constructs the BNN model from the architecture description.
    """
    def __init__(self, arch_desc, output_size):
        super(GenBNN, self).__init__()
        self.input_size = arch_desc.blocks[0].in_dim
        self.output_size = output_size
        self.num_internal_blocks = len(arch_desc.blocks)
        self.name = arch_desc.name
        layers = []

        for blk in arch_desc.blocks:
            fc = BinarizeLinear(blk.in_dim, blk.out_dim)
            layers.append(fc)
            bn = nn.BatchNorm1d(blk.out_dim)
            layers.append(bn)
            htanh = nn.Hardtanh()
            layers.append(htanh)

        fc = BinarizeLinear(arch_desc.blocks[-1].out_dim, self.output_size)
        layers.append(fc)
        self.binarize_func = Binarize

        self.drop = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax()
        self.mod_list = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.binarize_func(x)
        for layer in self.mod_list:
            x = layer(x)

        return self.logsoftmax(x)

class BNNModel(object):

    def factory(arch_type, input_size, output_size):
        input_size = input_size[0] * input_size[1]
        if arch_type == '1blk_20':
            return BNN_1blk_20(input_size, output_size)
        elif arch_type == '1blk_20_trojan':
            return BNN_1blk_20_trojan(input_size, output_size)
        elif arch_type == '1blk_50':
            return BNN_1blk_50(input_size, output_size)
        elif arch_type == '1blk_50_trojan':
            return BNN_1blk_50_trojan(input_size, output_size)
        elif arch_type == '1blk_100':
            return BNN_1blk_100(input_size, output_size)
        elif arch_type == '1blk_100_trojan':
            return BNN_1blk_100_trojan(input_size, output_size)
        elif arch_type == '1blk_150':
            return BNN_1blk_150(input_size, output_size)
        elif arch_type == '1blk_150_trojan':
            return BNN_1blk_150_trojan(input_size, output_size)
        elif arch_type == '1blk_200':
            return BNN_1blk_200(input_size, output_size)
        elif arch_type == '1blk_200_trojan':
            return BNN_1blk_200_trojan(input_size, output_size)
        elif arch_type == '1blk_250':
            return BNN_1blk_250(input_size, output_size)
        elif arch_type == '1blk_250_trojan':
            return BNN_1blk_250_trojan(input_size, output_size)
        elif arch_type == '1blk_300':
            return BNN_1blk_300(input_size, output_size)
        elif arch_type == '1blk_300_trojan':
            return BNN_1blk_300_trojan(input_size, output_size)
        elif arch_type == '2blk_25_10':
            return BNN_2blk_25_10(input_size, output_size)
        elif arch_type == '2blk_25_10_trojan':
            return BNN_2blk_25_10_trojan(input_size, output_size)
        elif arch_type == '2blk_50_20':
            return BNN_2blk_50_20(input_size, output_size)
        elif arch_type == '2blk_50_20_trojan':
            return BNN_2blk_50_20_trojan(input_size, output_size)
        elif arch_type == '2blk_100_50':
            return BNN_2blk_100_50(input_size, output_size)
        elif arch_type == '2blk_100_50_trojan':
            return BNN_2blk_100_50_trojan(input_size, output_size)
        elif arch_type == '2blk_100_100':
            return BNN_2blk_100_100(input_size, output_size)
        elif arch_type == '2blk_100_100_trojan':
            return BNN_2blk_100_100_trojan(input_size, output_size)
        elif arch_type == '2blk_150_100':
            return BNN_2blk_150_100(input_size, output_size)
        elif arch_type == '2blk_150_100_trojan':
            return BNN_2blk_150_100_trojan(input_size, output_size)
        elif arch_type == '2blk_150_150':
            return BNN_2blk_150_150(input_size, output_size)
        elif arch_type == '2blk_150_150_trojan':
            return BNN_2blk_150_150_trojan(input_size, output_size)
        elif arch_type == '2blk_200_100':
            return BNN_2blk_200_100(input_size, output_size)
        elif arch_type == '2blk_200_100_trojan':
            return BNN_2blk_200_100_trojan(input_size, output_size)
        elif arch_type == '2blk_200_150':
            return BNN_2blk_200_150(input_size, output_size)
        elif arch_type == '2blk_200_150_trojan':
            return BNN_2blk_200_150_trojan(input_size, output_size)
        elif arch_type == '2blk_200_200':
            return BNN_2blk_200_200(input_size, output_size)
        elif arch_type == '2blk_200_200_trojan':
            return BNN_2blk_200_200_trojan(input_size, output_size)
        elif arch_type == '3blk_200_100':
            return BNN_3blk_200_100(input_size, output_size)
        elif arch_type == '3blk_200_100_trojan':
            return BNN_3blk_200_100_trojan(input_size, output_size)
        elif arch_type == '4blk_200_100':
            return BNN_4blk_200_100(input_size, output_size)
        elif arch_type == '4blk_200_100_trojan':
            return BNN_4blk_200_100_trojan(input_size, output_size)

        raise AssertionError('{} BNN type unsupported'.format(arch_type))

    factory = staticmethod(factory)