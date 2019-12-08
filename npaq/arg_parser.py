#!/usr/bin/env python

import argparse
import os
import models
import utils
import torch
import datetime
import time
import shutil
import adversarial
import trojan_attack
import definitions
import subprocess
import differential_privacy
import bnn_dataset
import adv_train2
import properties
import trojan_attack_dtm
import mc

import numpy as np
from copy import copy
import stats
import adv_train3
import bnn_derand

import logging
logger = logging.getLogger(__name__)

arch_choices = ['1blk_20', '1blk_50', '1blk_100', '1blk_150', '1blk_200', '1blk_250', '1blk_300',
                '2blk_25_10', '2blk_50_20', '2blk_100_50', '2blk_100_100', '2blk_150_100',
                '2blk_150_150', '2blk_200_100', '2blk_200_150', '2blk_200_200',
                '3blk_200_100', '4blk_200_100']

DATASET_NAME_LIST = ['mnist', 'purchase', 'fake_mnist', 'hmnist', 'diamonds', 'beer',
                     'location', 'gss_onehot', 'adv_train',
                     'gss_normal', 'steak_onehot', 'SAC1', 'SAC2', 'SAC3',
                     'steak_normal', 'uci_adult', 'others']

def dp(args):
    # assign values to global variables
    definitions.GRADIENT_NORM_BOUND = args.grad_bound
    definitions.NOISE_SCALE = args.noise_scale
    definitions.GROUP_SIZE = args.batch_size

    train_stats = stats.RecordStats(args.stats)
    if args.arch == 'all':
        print('This option is not available in training menu yet.')
        return

    # TODO warn about overwriting, Y/N before proceeding

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    nn = differential_privacy.BNN(args, train_stats)
    logs_dir = os.path.join(definitions.ROOT_DIR, '..', 'logs')
    utils.ensure_dir(logs_dir)
    logs_dir = os.path.join(logs_dir, 'dp_%s' % args.dataset)
    utils.ensure_dir(logs_dir)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s \
                            [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=os.path.join(logs_dir, nn.name + \
                                              '_{}.log'.format(datetime.datetime.now())),
                        level=logging.DEBUG)
    logger.debug(args)
    nn.run_train()

def train_bnn(args):
    train_stats = stats.RecordStats(args.stats)
    if args.arch == 'all':
        print('This option is not available in training menu yet.')
        return

    # TODO warn about overwriting, Y/N before proceeding

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)
    logger.debug(args)
    if args.split_and_train:
        dataset = args.dataset
        dataset1, dataset2 = utils.dataset_split_names(dataset)
        args.dataset = dataset1
        nn1 = bnn_dataset.BNN(args, train_stats)
        utils.set_logger_prop(args.dataset, 'train', nn1.name)

        train_sampler1, train_sampler2 = bnn_dataset.mnist_custom_split()

        ckpt = nn1.split_and_train(train_sampler1)

        args.dataset = dataset2
        nn2 = bnn_dataset.BNN(args, train_stats)
        utils.set_logger_prop(args.dataset, 'train', nn2.name)

        nn2.split_and_train(train_sampler2, checkpoint=ckpt)
    elif args.adj_train >= 0:
        # train on the adjacent dataset
        args.dataset = utils.adj_dataset(args.dataset, args.adj_train)
        nn = bnn_dataset.BNN(args, train_stats)
        utils.set_logger_prop(args.dataset, 'train', nn.name)
        print('Train adjacent; spliting with {} seed'.format(args.adj_train))
        adj_train_sampler = bnn_dataset.mnist_adjacent_training(nn.filename,
                                                                random_seed=args.adj_train)

        ckpt = nn.split_and_train(adj_train_sampler)

        logger.debug('Training on {} done. Model saved in \
                     {}'.format(args.dataset, ckpt))

    elif args.derand > 0:
        nn = bnn_derand.BNN(args, replace_times=args.derand)
        utils.set_logger_prop(args.dataset, 'train-derand', nn.name)

        nn.run_train()
    else:
        nn = bnn_dataset.BNN(args, train_stats)
        utils.set_logger_prop(args.dataset, 'train', nn.name)
        nn.run_train()

    train_stats.dump()
    train_stats.pp()

def dtm_trojan_train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    nn = trojan_attack_dtm.BNN(args)


def start_adv_train2(args):
    adv_stats = stats.RecordStats(args.stats)
    if args.arch == 'all':
        print('This option is not available in training menu yet.')
        return

    # TODO warn about overwriting, Y/N before proceeding

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    origin_dataset = args.dataset
    args.dataset = 'adv_%s' % origin_dataset

    nn = bnn_dataset.BNN(args, adv_stats, trojan_flag=True)
    utils.set_logger_prop(origin_dataset, 'adv_train2', nn.name)

    # load model
    model_dir = os.path.join(definitions.TRAINED_MODELS_DIR, origin_dataset)
    if ',' in args.resize:
        temp = args.resize.split(',')
        resize = int(temp[0]) * int(temp[1])
    else:
        resize = int(args.resize)
    pt_path = os.path.join(model_dir, '%s-%d-bnn_%s.pt' % (origin_dataset, resize, args.arch))
    if os.path.exists(pt_path):
        nn.model.load_state_dict(torch.load(pt_path, map_location={'cuda:0': 'cpu'}))
        print('Loaed model from %s' % pt_path)
    else:
        raise Exception(
            'ERROR: There is no trained model with %s arch for %s dataset => %s' % (
                args.arch, args.dataset, pt_path
            ))
    logger.debug(args)
    nn.run_train()

def gen_adv_data(args):
    if args.arch == 'all':
        print('This option is not available in training menu yet.')
        return

    # TODO warn about overwriting, Y/N before proceeding

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    nn = adv_train2.AdvTrain(args)
    utils.set_logger_prop(args.dataset, 'adv_data', nn.name)
    logger.debug(args)
    nn.prepare_dataset(split_ratio=0.8)

def start_adv_train3(args):
    adv_stats = stats.RecordStats(args.stats)
    if args.arch == 'all':
        print('This option is not available in training menu yet.')
        return

    # TODO warn about overwriting, Y/N before proceeding

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    # origin_dataset = args.dataset
    # args.dataset = '%s_adv3' % origin_dataset

    nn = adv_train3.AdvTrain(args)
    utils.set_logger_prop(args.dataset, 'adv_train3', nn.name)

    logger.debug(args)
    nn.run_train()

def run_trojan_attack(args):
    trojan_stats = stats.RecordStats(args.stats)
    if ',' in args.resize:
        temp = args.resize.split(',')
        resize = int(temp[0]) * int(temp[1])
    else:
        resize = int(args.resize)
    if args.mask_path == '':
        mask_path = os.path.join(os.path.join(definitions.TROJAN_DIR, 'trojan_mask'),
                                 'mask_%s_%d.bin' % (args.dataset, resize))
        args.mask_path = mask_path
    if os.path.exists(args.mask_path):
        pass
    else:
        raise Exception('ERROR: Unknown mask path => %s' % args.mask_path)
    if args.split:
        args.dataset, _ = utils.dataset_split_names(args.dataset)

    definitions.TROJAN_RETRAIN_DATASET_DIR = os.path.join(
        os.path.join(definitions.DATA_PATH, 'trojan_data'),
        '%s_%d_%s' % (args.dataset, resize, args.arch)
    )
    definitions.TROJAN_ORIGIN_DATA_DIR = os.path.join(definitions.TROJAN_RETRAIN_DATASET_DIR,
                                                      'origin_data')
    utils.check_folder(definitions.TROJAN_ORIGIN_DATA_DIR)
    print('Dataset Folder: ', definitions.TROJAN_RETRAIN_DATASET_DIR)

    utils.set_logger_prop(args.dataset, 'trojan_train', 'bnn_%s_%d' % (args.arch, resize))
    model_path, data_folder = trojan_attack.prepare_trojan_attack(args)
    # model_path = '/mnt/storage/teo/npaq/ccs-submission/experiments/models/mnist/mnist-784-bnn_1blk_100.pt'
    print('=> Finish generating dataset!')

    # retrain the model
    utils.set_logger_prop(args.dataset, 'trojan_train',
                          'bnn_%s_%d-%s' % (args.arch, resize, os.path.basename(data_folder)))
    definitions.TRAINED_MODELS_DIR = os.path.join(definitions.TRAINED_MODELS_DIR, 'trojan_%s' %
                                                  args.dataset)
    utils.ensure_dir(definitions.TRAINED_MODELS_DIR)
    definitions.TRAINED_MODELS_CP_DIR = os.path.join(definitions.TRAINED_MODELS_CP_DIR,
                                                     'trojan_%s' % args.dataset)
    utils.ensure_dir(definitions.TRAINED_MODELS_CP_DIR)
    args.dataset = 'trojan_%s-%s' % (args.dataset, os.path.basename(data_folder))
    args.data_folder = data_folder
    if args.arch == 'all':
        print('This option is not available in training menu yet.')
        return

    # TODO warn about overwriting, Y/N before proceeding

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    definitions.TRAINED_MODELS_DIR = os.path.join(definitions.TRAINED_MODELS_DIR, args.dataset)
    utils.ensure_dir(definitions.TRAINED_MODELS_DIR)
    definitions.TRAINED_MODELS_CP_DIR = os.path.join(definitions.TRAINED_MODELS_CP_DIR,
                                                     args.dataset)
    utils.ensure_dir(definitions.TRAINED_MODELS_CP_DIR)

    nn = bnn_dataset.BNN(args, trojan_stats, trojan_flag=True)
    nn.model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))
    logger.debug(args)
    nn.run_train()

def prepare_trojan_data(args):
    if args.split:
        args.dataset, _ = utils.dataset_split_names(args.dataset)
    if ',' in args.resize:
        temp = args.resize.split(',')
        resize = int(temp[0]) * int(temp[1])
    else:
        resize = int(args.resize)
    trojan_folder = os.path.join(definitions.DATA_PATH, 'trojan_data')
    utils.ensure_dir(trojan_folder)
    definitions.TROJAN_RETRAIN_DATASET_DIR = os.path.join(trojan_folder,
                                                          '%s_%d_%s' % (args.dataset, resize,
                                                                        args.arch))
    utils.ensure_dir(definitions.TROJAN_RETRAIN_DATASET_DIR)
    print('Dataset Folder: ', definitions.TROJAN_RETRAIN_DATASET_DIR)

    # prepare dataset
    utils.set_logger_prop(args.dataset, 'trojan_data', 'bnn_%s_%d' % (args.arch, resize))
    my_trojan = trojan_attack.TrojanAttack(args)
    # my_trojan.prepare_benign_train_data()
    my_trojan.prepare_real_train_data()

def test_bnn(args):
    test_stats = stats.RecordStats(args.stats)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    nn = bnn_dataset.BNN(args, test_stats)
    logs_dir = os.path.join(definitions.ROOT_DIR, '..', 'logs')
    utils.ensure_dir(logs_dir)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s \
                        [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=os.path.join(logs_dir, nn.name + \
                                              'test_{}.log'.format(datetime.datetime.now())),
                        level=logging.DEBUG)
    logger.debug(args)
    nn.test()

    test_stats.dump()
    test_stats.pp()

def encode_bnn(args):
    nn = bnn_dataset.BNN(args, stats)
    logs_dir = os.path.join(definitions.ROOT_DIR, '..', 'logs')
    utils.ensure_dir(logs_dir)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s \
                        [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=os.path.join(logs_dir, nn.name +
                                              'enc_{}.log'.format(datetime.datetime.now())),
                        level=logging.DEBUG)

    logger.debug(args)

    nn.load_model(save=True)
    # bnn.load_model()
    start = time.time()
    encoder = mc.BNNConverter(args.encoder)
    formula_fname = encoder.encode([nn])
    end = time.time()
    logger.debug('Encoding took {} sec'.format(end - start))

    if args.sparse_thresh:
        sparse_nn = nn.sparsify(lambda tensor: tensor.apply_(lambda x: 0 if abs(x)
                                                           < args.sparse_thresh
                                                           else x).sign(),
                                args.sparse_thresh)

def parse_dissimilarity(args):
    nn1 = bnn_dataset.BNN(args, stats)

    nn1.load_model(save=True)
    if args.w_sparsified:
        nn2 = nn1.sparsify(lambda tensor: tensor.apply_(lambda x: 0 if abs(x)
                                                           < args.w_sparsified
                                                           else x).sign(),
                              args.w_sparsified)
    elif args.w_arch:
        args.arch = args.w_arch
        nn2 = bnn_dataset.BNN(args, stats)
        nn2.load_model(save=True)
    elif args.w_trojan:
        nn2 = nn1.trojaned()
        nn2.load_model(save=True)
    elif args.w_new_trojan:
        # FIXME
        # XXX Hardcoding shit like it's the end of the world
        target = 0
        replace = 1

        orig_dataset = args.dataset
        args.dataset = 'dtm_benign-mnist'
        nn1 = bnn_dataset.BNN(args, stats)
        nn1.load_model(save=True)

        args.dataset = orig_dataset
        nn2 = nn1.new_trojaned(target, replace)
        nn2.load_model(save=True)

        dataset1, dataset2 = utils.dataset_split_names(orig_dataset)
        resize=args.resize.split(',')
        size = str(int(resize[0]) * int(resize[1]))

        # TODO parse the mask like in TROJAN TARGET
        utils.set_logger_prop(args.dataset, 'trojan-dissimilarity', nn1.filename)
        logger.debug(args)

        with open(os.path.join(definitions.TROJAN_MASK, 'mask_' + orig_dataset + '_' + size + '.bin')) as f:
            temp = f.readlines()
        content = ''.join(temp)
        temp = [ord(i) for i in content]
        mask = copy(np.asarray(temp))
        print(mask)
        locations = np.where(mask == 1)[0]
        print(locations)

        img_filename = 'trojan_' + dataset1 + '-target_' + str(target) + '-' + size + '-bnn_' + args.arch + '.bin'
        constraints = 'trojan_' + dataset1 + '-target_' + str(target) + '-' + size + '-bnn_' + args.arch + '.txt'
        img = bnn_dataset.data_loader(os.path.join(definitions.TROJAN_IMGS, img_filename))
        trigger = mask * img
        print('trigger: ', trigger)
        trigger = trigger.flatten()
        constraints_f = os.path.join(definitions.TROJAN_IMGS, constraints)
        with open(constraints_f, 'w') as f:
            for loc in locations:
                if trigger[loc] < 0:
                    value = 0
                else:
                    value = 1
                f.write('{} {}\n'.format(loc, value))

        print('CONSTRAINTS FILE: {}'.format(constraints_f))

        properties.quantify_dissim(nn1, nn2,
                                   constraints_fname=os.path.abspath(constraints_f),
                                   just_encode=True,
                                   enc_strategy=args.encoder)
        return

    elif args.w_split:
        dataset1, dataset2 = utils.dataset_split_names(args.dataset)
        args.dataset = dataset1
        nn1 = bnn_dataset.BNN(args, stats)
        nn1.load_model(save=True)

        args.dataset = dataset2
        nn2 = bnn_dataset.BNN(args, stats)
        nn2.load_model(save=True)
    elif args.w_trojan_split:
        dataset1, _ = utils.dataset_split_names(args.dataset)

        args.dataset = dataset1
        nn1 = bnn_dataset.BNN(args, stats)
        nn1.load_model(save=True)

        nn2 = nn1.trojaned()
        nn2.load_model(save=True)
    else:
        print('There should not be another choice')
        exit(1)

    logs_dir = os.path.join(definitions.ROOT_DIR, '..', 'logs')
    utils.set_logger_prop(args.dataset, 'dissimilarity', nn1.filename)
    logger.debug(args)

    properties.quantify_dissim(nn1, nn2, enc_strategy=args.encoder)

def parse_robustness(args):
    if args.dataset != 'mnist' and args.dataset != 'hmnist':
        print('Robustness not supported for dataset {}'.format(args.dataset))
        exit(1)

    epoch=-1
    if args.w_adv_train:
        args.dataset = 'adv_train_' + args.dataset

    elif args.w_adv_train2 > 0:
        args.dataset = 'adv_' + args.dataset
        epoch = args.w_adv_train2
    elif args.w_adv_train3 > 0:
        args.dataset = 'adv_3_' + args.dataset
        epoch = args.w_adv_train3

    nn = bnn_dataset.BNN(args, stats, epoch=epoch)
    nn.load_model(save=True)
    utils.set_logger_prop(args.dataset, 'robustness', nn.filename)

    logger.debug(args)
    logger.debug('Quantify robustness')
    properties.quantify_robustness(nn, args.perturb, args.concrete_ip,
                                   dataset=args.dataset,
                                   equal=args.equal,
                                   num_samples=args.num_samples,
                                   just_encode=args.just_encode,
                                   enc_strategy=args.encoder)

def parse_label(args):
    if args.dataset != 'mnist':
        print('Label property is not supported for dataset {}'.format(args.dataset))
        exit(1)

    if args.w_trojan == 'nn1':
        dataset1, dataset2 = utils.dataset_split_names(args.dataset)

        # mnist 1
        args.dataset = dataset1
        nn1 = bnn_dataset.BNN(args, stats)
        nn1.load_model(save=True)

        utils.set_logger_prop(dataset1, 'labels_trojan', nn1.filename)
        logger.debug(args)
        logger.debug('Quantifying per-label distributions of mnist_1')
        properties.quantify_dp(nn1, just_encode=args.just_encode)
    elif args.w_trojan == 'trojaned':
        dataset1, dataset2 = utils.dataset_split_names(args.dataset)

        # mnist 1
        args.dataset = dataset1
        nn1 = bnn_dataset.BNN(args, stats)
        nn1.load_model(save=True)

        # trojaned mnist 1
        nn_trojaned = nn1.trojaned(args.target)
        nn_trojaned.load_model(save=True)

        utils.set_logger_prop(dataset1, 'labels_trojan', nn_trojaned.filename)
        logger.debug(args)
        logger.debug('Quantifying per-label distributions of trojan')
        properties.quantify_dp(nn_trojaned, just_encode=args.just_encode)

    elif args.w_trojan == 'nn2':
        dataset1, dataset2 = utils.dataset_split_names(args.dataset)
        # mnist 2
        args.dataset = dataset2
        nn2 = bnn_dataset.BNN(args, stats)
        nn2.load_model(save=True)

        utils.set_logger_prop(dataset2, 'labels_trojan', nn2.filename)
        logger.debug(args)
        logger.debug('Quantifying per-label distributions of mnist_2')

        properties.quantify_dp(nn2, just_encode=args.just_encode)
    elif args.w_trojan == 'trojan-success':
        orig_dataset = args.dataset
        dataset1, dataset2 = utils.dataset_split_names(orig_dataset)
        resize=args.resize.split(',')
        size = str(int(resize[0]) * int(resize[1]))

        utils.set_logger_prop(args.dataset, 'trojan_success', 'all-targets_0,1,4,5,9')
        logger.debug(args)

        for target in definitions.TROJAN_TARGETS:
            with open(os.path.join(definitions.TROJAN_MASK, 'mask_' + orig_dataset + '_' + size + '.bin')) as f:
                temp = f.readlines()
            content = ''.join(temp)
            temp = [ord(i) for i in content]
            mask = copy(np.asarray(temp))
            print(mask)
            locations = np.where(mask == 1)[0]
            print(locations)

            img_filename = 'trojan_' + dataset1 + '-target_' + str(target) + '-' + size + '-bnn_' + args.arch + '.bin'
            constraints = 'trojan_' + dataset1 + '-target_' + str(target) + '-' + size + '-bnn_' + args.arch + '.txt'
            img = bnn_dataset.data_loader(os.path.join(definitions.TROJAN_IMGS, img_filename))
            trigger = mask * img
            print('trigger: ', trigger)
            trigger = trigger.flatten()
            constraints_f = os.path.join(definitions.TROJAN_IMGS, constraints)
            with open(constraints_f, 'w') as f:
                for loc in locations:
                    if trigger[loc] < 0:
                        value = 0
                    else:
                        value = 1
                    f.write('{} {}\n'.format(loc, value))

            logger.debug('Quantifying per-label distributions for trigger {}'.format(os.path.abspath(constraints_f)))
            for epoch in definitions.TROJAN_EPOCHS:
                print('epoch: %d' % epoch)
                args.dataset = 'trojan_' + dataset1 + '-target_' + str(target)
                trojaned = bnn_dataset.BNN(args, stats, epoch=epoch)
                trojaned.load_model(save=True)
                print(trojaned.filename)

                properties.quantify_trojan_success(trojaned, target,
                                                   os.path.abspath(constraints_f),
                                                   just_encode=True,
                                                   enc_strategy=args.encoder)
    elif args.w_trojan == 'trojan-all-success':
        orig_dataset = args.dataset
        dataset1, dataset2 = utils.dataset_split_names(orig_dataset)
        resize=args.resize.split(',')
        size = str(int(resize[0]) * int(resize[1]))

        utils.set_logger_prop(args.dataset, 'trojan_success',
                              'all-targets_0,1,4,5,9-all-labels')
        logger.debug(args)

        for target in definitions.TROJAN_TARGETS:
            with open(os.path.join(definitions.TROJAN_MASK, 'mask_' + orig_dataset + '_' + size + '.bin')) as f:
                temp = f.readlines()
            content = ''.join(temp)
            temp = [ord(i) for i in content]
            mask = copy(np.asarray(temp))
            print(mask)
            locations = np.where(mask == 1)[0]
            print(locations)

            img_filename = 'trojan_' + dataset1 + '-target_' + str(target) + '-' + size + '-bnn_' + args.arch + '.bin'
            constraints = 'trojan_' + dataset1 + '-target_' + str(target) + '-' + size + '-bnn_' + args.arch + '.txt'
            img = bnn_dataset.data_loader(os.path.join(definitions.TROJAN_IMGS, img_filename))
            trigger = mask * img
            print('trigger: ', trigger)
            trigger = trigger.flatten()
            constraints_f = os.path.join(definitions.TROJAN_IMGS, constraints)
            with open(constraints_f, 'w') as f:
                for loc in locations:
                    if trigger[loc] < 0:
                        value = 0
                    else:
                        value = 1
                    f.write('{} {}\n'.format(loc, value))

            logger.debug('Quantifying per-label distributions for trigger {}'.format(os.path.abspath(constraints_f)))
            for epoch in definitions.TROJAN_EPOCHS:
                print('epoch: %d' % epoch)
                args.dataset = 'trojan_' + dataset1 + '-target_' + str(target)
                trojaned = bnn_dataset.BNN(args, stats, epoch=epoch)
                trojaned.load_model(save=True)
                print(trojaned.filename)

                for label in range(0, trojaned.num_classes):
                    properties.quantify_trojan_success(trojaned, label,
                                                       os.path.abspath(constraints_f),
                                                       just_encode=True,
                                                       enc_strategy=args.encoder)
    elif args.w_dp:
        nn = bnn_dataset.BNN(args, stats)
        nn.load_model(save=True)
        utils.set_logger_prop(args.dataset, 'dp', nn.filename)
        logger.debug(args)
        logger.debug('Quantifying per-label distributions')

        properties.quantify_dp(nn, just_encode=args.just_encode)
    elif args.w_adj_dp >= 0:
        args.dataset = utils.adj_dataset(args.dataset, args.w_adj_dp)
        adj_nn = bnn_dataset.BNN(args, stats)
        adj_nn.load_model(save=True)
        utils.set_logger_prop(args.dataset, 'dp', adj_nn.name)
        logger.debug(args)
        logger.debug('Quantifying per-label distributions')

        properties.quantify_dp(adj_nn, just_encode=args.just_encode)
    else:
        nn = bnn_dataset.BNN(args, stats)
        nn.load_model(save=True)

        utils.set_logger_prop(args.dataset, 'dp', nn.name)
        logger.debug(args)
        logger.debug('Quantifying per-label distributions')

        properties.quantify_dp(nn, just_encode=args.just_encode)


def parse_fair(args):
    nn = bnn_dataset.BNN(args, stats)
    nn.load_model(save=True)

    utils.set_logger_prop(args.dataset, 'fairness', nn.name)
    logger.debug(args)

    if args.dataset == 'uci_adult':
        dataset_ct_fname = definitions.UCI_CONSTRAINTS
    properties.quantify_fair(nn, args.constraints_fname,
                             dataset_ct_fname,
                             just_encode=args.just_encode,
                             enc_strategy=args.encoder)

def parse_canary(args):
    nn = bnn_derand.BNN(args, replace_times=args.replace_times)
    nn.load_model(save=True)

def create_parser():
    parser = argparse.ArgumentParser(description='nncrusher tool')
    #     parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                        # help='Run for a single instruction or for a batch of \
                        # instr')
    parser.add_argument('--results_dir', type=str,
                        help='Directory to save all the weights and biases of the \
                              model under results/model and the encoding into CNF \
                              under results/encoding. Default is results/.')
    parser.add_argument('--stats', type=str, default='stats.txt',
                        help='Collect stats in stats file')
    parser.add_argument('--debug', type=str, default=None, help='Enable debug \
                        prints to stdout. By default, not enabled.')
    # XXX remove if it's not used
    # parser.add_argument('--in_sample_file', type=str, help='File with input \
                        # samples.')

    subparsers = parser.add_subparsers(dest='subparser_name')

    # BNN Dataset
    parser_bnn = subparsers.add_parser('bnn', help='BNN MLP for a Given Dataset')
    parser_bnn.add_argument('--dataset', type=str, choices=DATASET_NAME_LIST,
                            help='Specify what dataset the BNN is trained on.')
    parser_bnn.add_argument('--data_folder', type=str, default='',
                            help='Specify the folder path where the dataset is saved. This \
                                 argument is necessary when --dataset != mnist')
    # can check if resize is present or not. If it is, then resize, otherwhise
    # keep original dataset size
    parser_bnn.add_argument('--resize', type=str, default='28,28',
                            help='Specify a resize of the MNIST PIL images as two \
                            numbers separated by comma, e.g. \'10,10\'. By default \
                            MNIST is 28,28. Some options allow --resize=all')
    parser_bnn.add_argument('--num_classes', type=int, default=10,
                            help='Specify the number of output classes the dataset has. By \
                                 default MNIST has 10 classes.')
    parser_bnn.add_argument('--arch', type=str, default='1blk_20',
                            choices=arch_choices,
                            help='Type of BNN architecture. Default is 1blk_20. If all \
                            is selected and action `encode` is selected then the \
                            encodes all the architectures of input size specified by \
                            --resize.')
    # XXX when we add this option for other types of networks like MLP, the options
    # should be tied to the parser
    parser_bnn.add_argument('--config', type=str, help='File containing BNN \
                                architecture description in JSON format.')
    parser_bnn.add_argument('--encoder', type=str, choices=['best', 'bdd', 'card'],
                            default='best', help='Select the encoder for \
                            encoding the BNN')
    subparsers = parser_bnn.add_subparsers()

    # Encode BNN
    parser_encode = subparsers.add_parser('encode')
    parser_encode.add_argument('--sparse_thresh', type=float,
                               help='Encode the sparsified model. The sparse model \
                               parameters are located at \
                               $dataset-$size-nn_name-sparse=$sparse_thresh.params')
    parser_encode.set_defaults(func=encode_bnn)

    # Train BNN
    parser_train = subparsers.add_parser('train')
    # parser_train.add_argument('--derand', action='store_true', help='Always \
                              # train the same thing.')
    parser_train.add_argument('--derand', type=int, default=-1,
                              help='Replace with canary and start from the same \
                              model with same seed in training.')
    parser_train.add_argument('--split-and-train', action='store_true',
                              help='Split the dataset, train on the first part \
                              of the dataset, save and train on the rest of the \
                              dataset and save.')
    parser_train.add_argument('--adj-train', type=int,
                              help='Create an adjacent training sets with a \
                              fixed seed specified by --ad-train and train on \
                              that adjacent training \ set.')
    parser_train.add_argument('--batch-size', type=int, default=256, metavar='N',
                              help='input batch size for training (default: 256)')
    parser_train.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
    parser_train.add_argument('--epochs', type=int, default=100, metavar='N',
                              help='number of epochs to train (default: 100)')
    parser_train.add_argument('--early-stop', type=int, default=10, metavar='ES',
                              help='number of epochs to wait if the test loss does not decrease \
                                    (default: 10). If < 0, then no early stopping.')
    parser_train.add_argument('--lr', type=float, default=0.001, metavar='LR',
                              help='learning rate (default: 0.001)')
    parser_train.add_argument('--momentum', type=float, default=0.5, metavar='M',
                              help='SGD momentum (default: 0.5)')
    parser_train.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_train.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_train.add_argument('--gpus', default=3,
                              help='gpus used for training - e.g 0,1,3')
    parser_train.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_train.add_argument('--save-interval', type=int, default=1, metavar='SI',
                              help='how many epochs to wait before saving the model (default: 1)')
    parser_train.set_defaults(func=train_bnn)

    # For generating adversarial samples
    parser_adv_data = subparsers.add_parser('adv_data')
    parser_adv_data.add_argument('--max_change', type=int, default=3, metavar='MC',
                                  help='The maximum number of bits allowed for changing. (\
                                           Default: 3)')
    parser_adv_data.add_argument('--batch-size', type=int, default=256, metavar='N',
                                  help='input batch size for training (default: 256)')
    parser_adv_data.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                                  help='input batch size for testing (default: 1000)')
    parser_adv_data.add_argument('--no-cuda', action='store_true', default=False,
                                  help='disables CUDA training')
    parser_adv_data.add_argument('--seed', type=int, default=1, metavar='S',
                                  help='random seed (default: 1)')
    parser_adv_data.set_defaults(func=gen_adv_data)

    # For adversarial training 2
    parser_adv_train = subparsers.add_parser('adv_train')
    parser_adv_train.add_argument('--split-and-train', action='store_true',
                              help='Split the dataset, train on the first part \
                                  of the dataset, save and train on the rest of the \
                                  dataset and save.')
    parser_adv_train.add_argument('--adj-train', type=int,
                              help='Create an adjacent training sets with a \
                                  fixed seed specified by --ad-train and train on \
                                  that adjacent training \ set.')
    parser_adv_train.add_argument('--batch-size', type=int, default=256, metavar='N',
                              help='input batch size for training (default: 256)')
    parser_adv_train.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
    parser_adv_train.add_argument('--epochs', type=int, default=100, metavar='N',
                              help='number of epochs to train (default: 100)')
    parser_adv_train.add_argument('--early-stop', type=int, default=10, metavar='ES',
                              help='number of epochs to wait if the test loss does not decrease \
                                        (default: 10). If < 0, then no early stopping.')
    parser_adv_train.add_argument('--lr', type=float, default=0.001, metavar='LR',
                              help='learning rate (default: 0.001)')
    parser_adv_train.add_argument('--momentum', type=float, default=0.5, metavar='M',
                              help='SGD momentum (default: 0.5)')
    parser_adv_train.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_adv_train.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_adv_train.add_argument('--gpus', default=3,
                              help='gpus used for training - e.g 0,1,3')
    parser_adv_train.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_adv_train.add_argument('--save-interval', type=int, default=1, metavar='SI',
                              help='how many epochs to wait before saving the model (default: 1)')
    parser_adv_train.set_defaults(func=start_adv_train2)

    # For adversarial training 3
    parser_adv_train3 = subparsers.add_parser('adv_train3')
    parser_adv_train3.add_argument('--max_change', type=int, default=3, metavar='MC',
                                 help='The maximum number of bits allowed for changing. (\
                                               Default: 3)')
    parser_adv_train3.add_argument('--split-and-train', action='store_true',
                                  help='Split the dataset, train on the first part \
                                      of the dataset, save and train on the rest of the \
                                      dataset and save.')
    parser_adv_train3.add_argument('--adj-train', type=int,
                                  help='Create an adjacent training sets with a \
                                      fixed seed specified by --ad-train and train on \
                                      that adjacent training \ set.')
    parser_adv_train3.add_argument('--batch-size', type=int, default=256, metavar='N',
                                  help='input batch size for training (default: 256)')
    parser_adv_train3.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                                  help='input batch size for testing (default: 1000)')
    parser_adv_train3.add_argument('--epochs', type=int, default=100, metavar='N',
                                  help='number of epochs to train (default: 100)')
    parser_adv_train3.add_argument('--early-stop', type=int, default=10, metavar='ES',
                                  help='number of epochs to wait if the test loss does not decrease \
                                            (default: 10). If < 0, then no early stopping.')
    parser_adv_train3.add_argument('--lr', type=float, default=0.001, metavar='LR',
                                  help='learning rate (default: 0.001)')
    parser_adv_train3.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                  help='SGD momentum (default: 0.5)')
    parser_adv_train3.add_argument('--no-cuda', action='store_true', default=False,
                                  help='disables CUDA training')
    parser_adv_train3.add_argument('--seed', type=int, default=1, metavar='S',
                                  help='random seed (default: 1)')
    parser_adv_train3.add_argument('--gpus', default=3,
                                  help='gpus used for training - e.g 0,1,3')
    parser_adv_train3.add_argument('--log-interval', type=int, default=10, metavar='N',
                                  help='how many batches to wait before logging training status')
    parser_adv_train3.add_argument('--save-interval', type=int, default=1, metavar='SI',
                                  help='how many epochs to wait before saving the model (default: 1)')
    parser_adv_train3.set_defaults(func=start_adv_train3)


    # prepare training dataset for trojan attack
    parser_trojan_data = subparsers.add_parser('trojan_data')
    parser_trojan_data.add_argument('--verbose', type=bool, default=False, metavar='V',
                               help='Save all the temperal information if --verbose=True (\
                                             default: False).')
    parser_trojan_data.add_argument('--data_size', type=int, default=100,
                               help='Specify the number of retrain data for all the classes. (Default: 100)')
    parser_trojan_data.add_argument('--train_data_threshold', type=float, default=0.0001,
                               help='Specify the maximum threshold which can be tolerated during \
                                        when generating retraining dataset. (Default: 0.0001)')
    parser_trojan_data.add_argument('--train_data_epoch', type=int, default=1000,
                               help='Specify the number of epochs for generating one retrain \
                                        sample. (Default: 1000)')
    parser_trojan_data.add_argument('--split', action='store_true', help='Attack on \
                                   the split dataset model.')
    parser_trojan_data.set_defaults(func=prepare_trojan_data)

    # deterministic trojan attack on BNN
    parser_dtm_trojan = subparsers.add_parser('dtm_trojan')
    parser_dtm_trojan.add_argument('--batch-size', type=int, default=256, metavar='N',
                              help='input batch size for training (default: 256)')
    parser_dtm_trojan.add_argument('--target-class', type=str, default='0', metavar='TC',
                                   help='The target class of trojan attack. (default: 0)')
    parser_dtm_trojan.add_argument('--selected-neuron', type=str, default='1_16', metavar='SN',
                                   help='The target neuron which is activated for trojan attack. \
                                   (default: 1_16)')
    parser_dtm_trojan.add_argument('--replace-times', type=int, default=1, metavar='RT',
                                   help='The number of trojaned inputs injected in the training \
                                   set. (default: 1)')
    parser_dtm_trojan.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
    parser_dtm_trojan.add_argument('--epochs', type=int, default=100, metavar='N',
                              help='number of epochs to train (default: 100)')
    parser_dtm_trojan.add_argument('--lr', type=float, default=0.001, metavar='LR',
                              help='learning rate (default: 0.001)')
    parser_dtm_trojan.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_dtm_trojan.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_dtm_trojan.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_dtm_trojan.add_argument('--save-interval', type=int, default=1, metavar='SI',
                              help='how many epochs to wait before saving the model (default: 1)')
    parser_dtm_trojan.set_defaults(func=dtm_trojan_train)

    # Trojan attack on BNN
    parser_trojan = subparsers.add_parser('trojan')
    parser_trojan.add_argument('--verbose', type=bool, default=False, metavar='V',
                               help='Save all the temperal information if --verbose=True (\
                                         default: False).')
    parser_trojan.add_argument('--mask_path', type=str, default='',
                               help='Specify the path for mask.')
    parser_trojan.add_argument('--target_class', type=int, default=1,
                               help='Specify the target output class for the trigger. (Default: 1)')
    parser_trojan.add_argument('--split', action='store_true', help='Attack on \
                               the split dataset model.')
    parser_trojan.add_argument('--balance_strategy', type=str,
                               choices=['more_benign', 'no_balance'],
                               default='no_balance', help='Specify the strategy used for \
                               balancing the retraining dataset. (Default: no_balance)')
    # For selecting neurons
    parser_trojan.add_argument('--layers', type=str, default='1', metavar='L',
                               help='Specify the target layers for selecting the target neurons (\
                                    default: 1 (first layer)). If there are multiple layers, \
                                    please use "," to separate the layers. The no of output \
                                    layer is 5.')
    parser_trojan.add_argument('--neuron_num', type=int, default=1, metavar='N',
                               help='Specify the number of target neurons the attack aims at (\
                                         default: 1).')
    parser_trojan.add_argument('--select_strategy', type=str, choices=['random', 'real_weights',
                                                                       'user_defined'],
                               default='random', metavar='S',
                               help='Specify the strategy used for selecting neurons. (default: \
                               real_weights).')
    parser_trojan.add_argument('--neuron', type=str, help='Specify the selected neurons when \
                                                          --select_strategy=user_defined')
    # For generating triggers
    parser_trojan.add_argument('--trigger_threshold', type=float, default=0.0001,
                               help='Specify the maximum threshold which can be tolerated during \
                                    when generating trigger. (Default: 0.0001)')
    parser_trojan.add_argument('--trigger_epoch', type=int, default=1000,
                               help='Specify the number of epochs for generating the trigger. (\
                                    Default: 1000)')
    parser_trojan.add_argument('--target_value', type=float, default=2.0,
                               help='Specify the target value the user wants the selected neuron \
                                    to achieve. (Default: 2.0)')
    # For retraining
    parser_trojan.add_argument('--batch-size', type=int, default=256, metavar='N',
                              help='input batch size for training (default: 256)')
    parser_trojan.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
    parser_trojan.add_argument('--epochs', type=int, default=100, metavar='N',
                              help='number of epochs to train (default: 100)')
    parser_trojan.add_argument('--early-stop', type=int, default=10, metavar='ES',
                              help='number of epochs to wait if the test loss does not decrease \
                                        (default: 10). If < 0, then no early stopping.')
    parser_trojan.add_argument('--lr', type=float, default=0.001, metavar='LR',
                              help='learning rate (default: 0.001)')
    parser_trojan.add_argument('--momentum', type=float, default=0.5, metavar='M',
                              help='SGD momentum (default: 0.5)')
    parser_trojan.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_trojan.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_trojan.add_argument('--gpus', default=3,
                              help='gpus used for training - e.g 0,1,3')
    parser_trojan.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_trojan.add_argument('--save-interval', type=int, default=1, metavar='SI',
                              help='how many epochs to wait before saving the model (default: 1)')

    parser_trojan.set_defaults(func=run_trojan_attack)

    # Train Differential Private BNN
    parser_dp = subparsers.add_parser('dp')
    parser_dp.add_argument('--verbose', type=bool, default=False,
                           help='Determine whether outputing the intermediate info. (Default: \
                                False)')
    parser_dp.add_argument('--grad-bound', type=float, default=10.0,
                           help='The clipping norm to apply to the global norm of each record. (\
                                Default: 10.0)')
    parser_dp.add_argument('--delta', type=float, default=0.00001,
                           help='The delta used in differential privacy. (Default: 0.00001)')
    parser_dp.add_argument('--noise-scale', type=float, default=2.0,
                           help='The stddev of the noise added to the sum. (Default: 2.0)')
    parser_dp.add_argument('--target-eps', type=float, default=0.5,
                           help='The target eps to achieve. Once it reaches the target eps or \
                                above it, the training procedure will be stopped. If the target \
                                eps is smaller than 0, the training procedure will disable the \
                                stopping mechanism based on eps. (Default: 0.5)')
    # Also including in original training (part of BP-BNN)
    parser_dp.add_argument('--batch-size', type=int, default=256, metavar='N',
                              help='input batch size for training (default: 256)')
    parser_dp.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
    parser_dp.add_argument('--epochs', type=int, default=100, metavar='N',
                              help='number of epochs to train (default: 100)')
    # parser_dp.add_argument('--early-stop', type=int, default=10, metavar='ES',
    #                           help='number of epochs to wait if the test loss does not decrease \
    #                                     (default: 10). If < 0, then no early stopping.')
    parser_dp.add_argument('--lr', type=float, default=0.001, metavar='LR',
                              help='learning rate (default: 0.001)')
    parser_dp.add_argument('--momentum', type=float, default=0.5, metavar='M',
                              help='SGD momentum (default: 0.5)')
    parser_dp.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_dp.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_dp.add_argument('--gpus', default=3,
                              help='gpus used for training - e.g 0,1,3')
    parser_dp.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_dp.add_argument('--save-interval', type=int, default=1, metavar='SI',
                              help='how many epochs to wait before saving the model (default: 1)')
    parser_dp.set_defaults(func=dp)


    # Test BNN
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--sparse_thresh', type=float,
                             help='Test the sparsified model that is in \
                             $results_dir/$nn_name/model_sparse_$thresh.')
    parser_test.add_argument('--batch-size', type=int, default=64, metavar='N',
                              help='input batch size for training (default: 256)')
    parser_test.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
    parser_test.add_argument('--lr', type=float, default=0.001, metavar='LR',
                              help='learning rate (default: 0.001)')
    parser_test.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_test.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_test.add_argument('--gpus', default=3,
                              help='gpus used for training - e.g 0,1,3')
    parser_test.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_test.set_defaults(func=test_bnn)

    # Dis-similarity
    parser_dissim = subparsers.add_parser('quant-dis-sim')
    parser_dissim.add_argument('--w_sparsified', type=float,
                              help='Encode and compute the dis-similarity \
                              property of the original model and the model \
                              sparsfied with $w-sparsified threshold.')
    parser_dissim.add_argument('--w_arch', type=str, choices=arch_choices,
                               help='Encode and compute the dis-similarity of \
                               --arch <model> with --w_arch <model>.')
    parser_dissim.add_argument('--w_trojan', action='store_true',
                               help='Encode and compute the dis-similarity \
                               between the model before and after the trojan \
                               poisoning.')
    parser_dissim.add_argument('--w_new_trojan', action='store_true',
                               help='Encode and compute the dis-similarity \
                               between the model before and after the trojan \
                               poisoning.')
    parser_dissim.add_argument('--w_split', action='store_true',
                               help='Difference between trained models on the \
                               split datasets.')
    parser_dissim.add_argument('--w_trojan_split', action='store_true',
                               help='Difference between trojan and original \
                               with split dataset.')
    parser_dissim.set_defaults(func=parse_dissimilarity)

    # Robustness
    parser_robustness = subparsers.add_parser('quant-robust')
    parser_robustness.add_argument('perturb', type=str,
                                   help='Encode and compute the robustness \
                                   property for changes in the input with at \
                                   most perturb bits.')
    parser_robustness.add_argument('--num_samples', type=int, default=1,
                                   help='Number of \
                                   samples for which to calculate robustenss.')
    parser_robustness.add_argument('--concrete_ip', type=str,
                                   help='Optionally, can give path to an \
                                   input. For each dataset, a different format \
                                   is expected.')
    parser_robustness.add_argument('--equal', action='store_true',
                                   default=False, help='Exactly perturb \
                                   perturbations on the input are allowed')
    parser_robustness.add_argument('--w-adv-train', action='store_true',
                                   help='Quantify robustness of adversarially \
                                   trained model.')
    parser_robustness.add_argument('--w-adv-train2', type=int, default=-1,
                                   help='Quantify robustness of adversarially \
                                   trained model with METHOD 2 at a certain \
                                   epoch.')
    parser_robustness.add_argument('--w-adv-train3', type=int, default=-1,
                                   help='Quantify robustness of adversarially \
                                   trained model with METHOD 2 at a certain \
                                   epoch.')
    parser_robustness.add_argument('--just-encode', action='store_true',
                                   help='DO not count, just encode.')
    parser_robustness.set_defaults(func=parse_robustness)

    # Per label counts
    parser_label = subparsers.add_parser('quant-label')
    parser_label.add_argument('--w_trojan', type=str,
                              choices=['nn1', 'trojaned','nn2',
                                       'trojan-success', 'trojan-all-success'],
                              help='Count probability per each output label for \
                              nn1: benign neural net, trojaned: after \
                              doing a trojan attack on the neural net \
                              and nn2: after more benign training.\
                              If trojan-success is selected, we compute the \
                              success of trigger for 3 epochs: 1, 10, 30 and \
                              5 target classes=[0,1,4,5,9] for the \
                              dataset-trojan_target-size-arch.')
    parser_label.add_argument('--target', type=str, default='0',
                              help='Target of the trojan attack. Default is 0.')
    parser_label.add_argument('--w_dp', action='store_true',
                              help='Count probability per each each output label \
                              for differentially private training with two \
                              adjacent training datasets d and d\'.')
    parser_label.add_argument('--w_adj_dp', type=int,
                              help='Count probability per each each output label \
                              for the model trained on the adjacent training \
                              datasets d\'. Specify seed of the randomly removed \
                              sample.')
    parser_label.add_argument('--just-encode', action='store_true',
                              help='DO not count, just encode.')
    parser_label.set_defaults(func=parse_label)

    parser_fairness = subparsers.add_parser('quant-fair')
    parser_fairness.add_argument('constraints_fname', type=str, help='Path to constrains file')
    parser_fairness.add_argument('--just-encode', action='store_true',
                                 help='DO not count, just encode.')
    parser_fairness.set_defaults(func=parse_fair)

    parser_canary = subparsers.add_parser('quant-canary')
    parser_canary.add_argument('replace_times', type=int, help='Replace times')
    parser_canary.set_defaults(func=parse_canary)
    return parser
