#!/usr/bin/env python

from __future__ import print_function

import definitions
import utils
import os
import math
import logging

logger = logging.getLogger(__name__)

def save_witness(dataset, formula_fname, nn):

    _, samples_fname = utils.get_output_fnames(formula_fname)
    if not os.path.exists(samples_fname):
        return (None, None)

    # write images to mnist-samples
    fname = os.path.basename(formula_fname)
    # write samples for the BNN in here
    # as per dataloader code in bnn.py
    dir_samples = os.path.join(definitions.MNIST_SAMPLES, fname, 'test',
                               'class_0')
    utils.ensure_dir(dir_samples)

    if dataset == 'mnist':
        # write images in here
        dir_imgs = os.path.join(definitions.MNIST_SAMPLES, fname, 'imgs')
        utils.ensure_dir(dir_imgs)

        _save_mnist_for_dataloader(samples_fname, dir_samples, dir_imgs)

        dir_assum_for_sat = os.path.join(definitions.MNIST_SAMPLES, fname, 'test', 'assumptions')
        utils.ensure_dir(dir_assum_for_sat)
        _mnist_assumptions_for_testing(samples_fname, dir_assum_for_sat, nn)
    else:
        print('dataset {} not supported yet!'.format(dataset))
        exit(1)

    return (os.path.join(definitions.MNIST_SAMPLES, fname, 'test'), dir_imgs)

def _save_mnist_for_dataloader(samples_fname, output_dir, output_img_dir):
    with open(samples_fname, 'r') as f:
        sample_idx = 0
        for sample_raw in f:

            sample = [int(x) for x in sample_raw.split(':')[1].strip(' ').split(' ')][:-1]
            if not int(math.sqrt(len(sample))) == math.sqrt(len(sample)):
                print('Expecting MNIST image to be a square for samples in {}'.format(samples_fname))
                exit(1)

            bin_sample = [1 if int(i) > 0 else 0 for i in sample]

            sample_img_fname = os.path.join(output_img_dir, 'sample_' + str(sample_idx) + '.png')
            utils.save_img(bin_sample, sample_img_fname)

            bin_sample_fname = os.path.join(output_dir, str(sample_idx) + '.bin')
            utils.write_sample(bin_sample_fname, bin_sample)
            sample_idx += 1

def _mnist_assumptions_for_testing(samples_fname, output_dir, nn):

    size = nn.resize[0] * nn.resize[1]

    with open(samples_fname, 'r') as f:
        sample_idx = 0
        for sample_raw in f:

            sample = [int(x) for x in sample_raw.split(':')[1].strip(' ').split(' ')][:-1]
            if not int(math.sqrt(len(sample))) == math.sqrt(len(sample)):
                logger.debug('Expecting MNIST image to be a square for samples in {}'.format(samples_fname))
                exit(1)

            bin_in_vec = [1 if int(i) > 0 else 0 for i in sample]
            binarized_ip = [-1.0 if x == 0 else 1.0 for x in bin_in_vec]

            pred = nn.predict(binarized_ip)
            bin_out_vec = [0 if idx != pred[0] else 1 for idx in range(nn.args.num_classes)]
            #print(binarized_ip, pred, bin_out_vec)

            with open(os.path.join(output_dir, 'assum-' +
                                   str(sample_idx)), 'w') as assum_f:
                # input variables
                assum_f.write(' '.join(str(i) for i in range(1, size + 1)) +
                              '\n')
                # output variables
                assum_f.write(' '.join(str(i) for i in range(size + 1, size + 1 +
                                                       nn.args.num_classes))+
                              '\n')
                values = ' '.join(str(i) for i in bin_in_vec) + ' '
                values += ' '.join(str(i) for i in bin_out_vec) + '\n'
                assum_f.write(values)

            sample_idx += 1

