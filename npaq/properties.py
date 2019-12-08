#!/usr/bin/env python

from __future__ import print_function

import os
import subprocess
import definitions
import quantifier
import witness
import utils
import logging
import multiprocessing
import time
from mc import PropType
import mc
import csv

"""
For all these, we write the results to a directory
"""

logger = logging.getLogger(__name__)

# XXX make these cmd line options
epsilon = 0.8
delta = 0.2
# 8h timeout
timeout = 28800
n_proc = 10

def concrete_in_name(name, perturb_k, perturb_num):
    utils.ensure_dir(definitions.CONCRETE_IN_DIR)

    filename = '{}-perturb_{}-id_{}.txt'.format(name, perturb_k, perturb_num)
    return os.path.join(definitions.CONCRETE_IN_DIR, filename)

def parsed_results(name, perturb):
    utils.ensure_dir(definitions.COUNT_OUT_DIR)

    filename = '{}-perturb_{}.csv'.format(name, perturb)

    return os.path.join(definitions.COUNT_OUT_DIR, filename)

def dp_parsed_results(name):
    utils.ensure_dir(definitions.COUNT_OUT_DIR)

    filename = '{}-label.csv'.format(name)

    return os.path.join(definitions.COUNT_OUT_DIR, filename)

def quantify_dissim(nn1, nn2, constraints_fname='',
                    dataset='mnist', just_encode=True, enc_strategy='best'):
    """
    Quantify the dis-similarity between two neural nets.
    Generate witnesses.

    Given the same input x, how many outputs differ between the two binarized
    neural networs?

    """
    if constraints_fname:
        encoder = mc.BNNConverter(enc_strategy=enc_strategy)
        formula_fname = encoder.encode([nn1, nn2],
                                       args=[constraints_fname],
                                       prop_type=PropType.DISSIMILARITY)

#     if not constraints_fname:
        # encoder = mc.BNNConverter(enc_strategy=enc_strategy)
        # formula_fname = encoder.encode([nn1, nn2], prop_type=PropType.DISSIMILARITY)

        # nn1_size = nn1.resize[0] * nn1.resize[1]
        # nn2_size = nn2.resize[0] * nn2.resize[1]
        # if nn1_size != nn2_size:
            # logger.debug('{} {} different sizes'.format(nn1_size, nn2_size))
            # exit(1)

        # quantifier.invoke_scalmc(formula_fname, nn1_size, epsilon, delta)
        # dir_samples, _ = witness.save_witness(dataset, formula_fname, nn1)
        # # TODO i also have predict function - maybe just use that?
        # nn1.predict_tandem(dir_samples, nn2)

def quantify_robustness(nn, perturb, concrete_ip, dataset='mnist',
                        enc_strategy='best', equal=False, num_samples=1,
                        just_encode=False):
    """
    Quantify robustness of a neural net.

    Given a concrete input from the holdout set, how many outputs within a
    bit-length hamming distance are classified with a different class?
    """
    nn_size = nn.resize[0] * nn.resize[1]
    if not concrete_ip:
        logger.debug('Pick randomly a sample correcty classified from the test set.')
        samples_pool = nn.pick_correct_test_samples(num_samples)

        start_time = time.time()
        for idx, (sample, _) in enumerate(samples_pool):
            concrete_str = ' '.join('1' if x > 0 else '0' for x in sample) + '\n'
            concrete_ip_fname = concrete_in_name(nn.filename, perturb, idx)
            with open(concrete_ip_fname, 'w') as f:
                f.write(concrete_str)

        logger.debug("Generating %s concrete inputs from the test set took %s \
                     seconds" % (num_samples, time.time() - start_time))
    else:
        if not os.path.exists(concrete_ip):
            logger.debug('concrete_ip does not exist - {}'.format(concrete_ip))
            exit(1)

        if os.path.isfile(concrete_ip):
            logger.debug('{} is a file. num_samples should be 1'.format(concrete_ip))
            if num_samples != 1:
                exit(1)

            # TODO do some other stuff
        elif os.path.isdir(concrete_ip):
            logger.debug('Reading concrete inputs from {}'.format(concrete_ip))
            logger.debug('Expecting \'dataset-size-name-*-id_*.txt\' filename format')

            filenames = os.listdir(concrete_ip)
            concrete_in_files = {}
            for fname in filenames:
                comps = os.path.splitext(fname)[0].split('-')
                # XXX if an adversarially trained model then we will load the same
                # concrete inputs as the model so we make this check on the last
                # 5 characters - a bit hackish
                if 'adv' in dataset and comps[0] == dataset[-5:] and int(comps[1]) == nn_size and comps[2] == nn.name:
                    idx = int(comps[4].split('_')[1])
                    if not idx in concrete_in_files:
                        concrete_in_files[idx] = os.path.join(concrete_ip,
                                                              fname)
                elif comps[0] == dataset and int(comps[1]) == nn_size and \
                        comps[2] == nn.name:
                    idx = int(comps[4].split('_')[1])
                    if not idx in concrete_in_files:
                        concrete_in_files[idx] = os.path.join(concrete_ip,
                                                              fname)

            logger.debug('Concrete_in_files {}'.format(concrete_in_files))

    start_time = time.time()
    tasks = []
    encoder = mc.BNNConverter(enc_strategy=enc_strategy)
    for idx in range(num_samples):
        if not concrete_ip:
            concrete_ip_fname = concrete_in_name(nn.filename, perturb, idx)
        else:
            concrete_ip_fname = concrete_in_files[idx]
        formula_fname = encoder.encode([nn],
                                       args=[perturb, concrete_ip_fname, equal,
                                             idx],
                                       prop_type=PropType.ROBUSTNESS)
        tasks.append((formula_fname, nn_size, epsilon, delta, timeout))

    logger.debug('Are we just encoding? {}'.format(just_encode))
    if just_encode:
        return

    logger.debug('Creating pool with %d processes\n' % n_proc)
    pool = multiprocessing.Pool(n_proc)
    logger.debug('pool = %s' % pool)

    results = [pool.apply_async(quantifier.invoke_scalmc, t) for t in tasks]
    res_fname = parsed_results(nn.filename, perturb)

    res_file = open(res_fname, 'w')
    reswriter = csv.writer(res_file, delimiter=',')

    counting_results = []
    for r in results:
        print(r.get())
        counting_results.append(r.get())
        reswriter.writerow(r.get())

    res_file.close()
    pool.close()
    pool.join()

    print('Counting done. Saving witnesses...')
    # Save witnesses generated by scalmc in the format that can be loaded by the
    # neural net test_loader
    for idx, t in enumerate(counting_results):
        formula_fname = t[0]
        if 'unsat' in t[1] or 'timeout' in t[1]:
            logger.debug('{} is {}'.format(t[0], t[1]))
            continue

        dir_samples, dir_imgs = witness.save_witness(dataset, formula_fname, nn)
        if not dir_samples or not dir_imgs:
            logger.debug('No samples to load for {}.'.format(formula_fname))
            continue

        # Save generated/user-provided concrete_input as image and log the
        # predictions for the witnesses generated by scalmc
        concrete_ip_fname = concrete_in_name(nn.filename, perturb, idx)
        logger.debug('read concrete ip from {}'.format(concrete_ip_fname))
        with open(concrete_ip_fname) as f:
            concrete_ip = [-1.0 if x == '0' else 1.0 for x in
                           f.readline().rstrip('\n').split(' ')]

            pred = nn.predict(concrete_ip)
            utils.save_img(concrete_ip, os.path.join(dir_imgs, 'concrete_ip.png'))
            logger.debug('[{}]: Input {} - Pred {}'.format(concrete_ip_fname,
                                                           concrete_ip, pred))
            nn.predict_samples(dir_samples)

    logger.debug("Robustness for %s concrete inputs from the test set took %s \
                 seconds" % (num_samples, time.time() - start_time))

def quantify_dp(nn, dataset='mnist', enc_strategy='best', just_encode=False):
    """
    Quantify differential privacy of the neural net.
    """
    start_time = time.time()
    encoder = mc.BNNConverter(enc_strategy=enc_strategy)

    tasks = []
    # encode and quantify
    for label in range(nn.num_classes):
        formula_fname = encoder.encode([nn], args=[label], prop_type=PropType.DP)
        tasks.append((formula_fname, label, epsilon, delta, timeout))

    logger.debug('Encoding {} formulas took {} seconds'.format(nn.num_classes,
                                                               time.time() -
                                                               start_time))
    logger.debug('Are we just encoding? {}'.format(just_encode))
    if just_encode:
        # for task in tasks:
            # formula_fname = task[0]
            # formula_fname_gz = formula_fname + '.gz'
        #     utils.gzip_formula(formula_fname, formula_fname_gz)
        return

    nn_size = nn.resize[0] * nn.resize[1]
    logger.debug('Creating pool with %d processes\n' % n_proc)
    pool = multiprocessing.Pool(n_proc)
    logger.debug('pool = %s' % pool)

    results = [pool.apply_async(quantifier.invoke_scalmc, t) for t in tasks]
    res_fname = dp_parsed_results(nn.filename)

    res_file = open(res_fname, 'w')
    reswriter = csv.writer(res_file, delimiter=',')

    counting_results = []
    for r in results:
        print(r.get())
        counting_results.append(r.get())
        reswriter.writerow(r.get())

    res_file.close()
    pool.close()
    pool.join()

    print('Counting done. Saving witnesses...')
    # Save witnesses generated by scalmc in the format that can be loaded by the
    # neural net test_loader
    for idx, t in enumerate(counting_results):
        formula_fname = t[0]
        if 'unsat' in t[1] or 'timeout' in t[1]:
            logger.debug('{} is {}'.format(t[0], t[1]))
            continue

        dir_samples, dir_imgs = witness.save_witness(dataset, formula_fname, nn)
        if not dir_samples or not dir_imgs:
            logger.debug('No samples to load for {}.'.format(formula_fname))
            continue

        # predictions for the witnesses generated by scalmc
        nn.predict_samples(dir_samples)

    logger.debug("Per label count for %s took %s seconds" % (nn.filename,
                                                             time.time() -
                                                             start_time))


def quantify_trojan_success(nn, label, constraints_fname, just_encode=False,
                            enc_strategy='best'):
    start_time = time.time()
    encoder = mc.BNNConverter(enc_strategy=enc_strategy)

    # encode and quantify
    formula_fname = encoder.encode([nn], args=[label, constraints_fname],
                                   prop_type=PropType.TROJAN)

    logger.debug('Encoding {} formulas took {} seconds'.format(nn.num_classes,
                                                               time.time() -
                                                               start_time))
    logger.debug('Are we just encoding? {}'.format(just_encode))
    if just_encode:
        # for task in tasks:
            # formula_fname = task[0]
            # formula_fname_gz = formula_fname + '.gz'
        #     utils.gzip_formula(formula_fname, formula_fname_gz)
        return

    # TODO this just has the encoding

def quantify_fair(nn, constraints_fname, dataset_ct_fname, enc_strategy='best',
                  just_encode=False):
    start_time = time.time()
    encoder = mc.BNNConverter(enc_strategy=enc_strategy)

    # encode and quantify
    formula_fname = encoder.encode([nn], args=[constraints_fname,
                                               dataset_ct_fname],
                                   prop_type=PropType.FAIR)

    logger.debug('Encoding {} formulas took {} seconds'.format(nn.num_classes,
                                                               time.time() -
                                                               start_time))
    logger.debug('Are we just encoding? {}'.format(just_encode))

