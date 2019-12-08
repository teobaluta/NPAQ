#!/usr/bin/env python

from __future__ import print_function

import unittest
import os
import bnn_dataset
import arg_parser
import stats
import pickle
import mc
import utils
import definitions

from multiprocessing import Pool

stats = stats.RecordStats('tests_stats.txt')
dataset = 'mnist'



# enumerate all bit vector of width
def gen_all_binvec(width):
    if width > 20:
        proceed = utils.query_yes_no('You are generating a {} (large amount) of \
                                     samples. Proceed?'.format(2**width))
        if not proceed:
            return []

    all_bin_vecs = []
    print(width)
    for num in range(2**width):
        bin_vec = width * [0]
        for i, bit in enumerate(bin(num)[2:][::-1]):
            bin_vec[i] = 0 if bit == '0' else 1

        all_bin_vecs.append(bin_vec)

    return all_bin_vecs

def run_bnn_cfg(v):
    args = v[0]
    stats = v[1]
    nn = bnn_dataset.BNN(args, stats)
    nn.trained_models_dir = os.path.join(definitions.ROOT_BNN, nn.filename, 'train')
    nn.saved_model = os.path.join(nn.trained_models_dir, nn.filename + '.pt')

    size = nn.resize[0] * nn.resize[1]
    inputs_pkl = os.path.join(definitions.TEST_SAMPLES_DIR, str(size) +
                              '-test_inputs.pkl')

    if not os.path.exists(inputs_pkl):
        inputs = gen_all_binvec(size)
        #print(inputs)
    else:
        print('{} exists. Not generating it again. Please remove it \
              manually.'.format(inputs_pkl))

        with open(inputs_pkl, 'rb') as f:
            inputs = pickle.load(f)

    samples = []
    for bin_in_vec in inputs:
        binarized_ip = [-1.0 if x == 0 else 1.0 for x in bin_in_vec]
        pred = nn.predict(binarized_ip)
        bin_out_vec = [0 if idx != pred[0] else 1 for idx in range(args.num_classes)]
        #print(binarized_ip, pred, bin_out_vec)
        samples.append((bin_in_vec, bin_out_vec))

    encoder = mc.BNNConverter('best')
    formula_fname = encoder.encode([nn])

    #output_filename = os.path.join(definitions.TEST_FORMULAS_DIR, args.encoder +
    #                               '-' + nn.filename + '-test.dimacs')
    #converter = mc.BNNConverter(nn.model_dir, output_filename)
    #print('config - {}; test encoding of CNF \
    #      {}'.format(os.path.basename(args.config), output_filename))
    #accuracy = converter.test_enc(samples, output_filename)
    accuracy = encoder.test_enc(samples, formula_fname)
    print('output {} - acc {}'.format(formula_fname, accuracy))

    return (args.config, accuracy)

class TestEnc(unittest.TestCase):

    def setUp(self):
        # TODO cleanup the tests_input directories - not sure if I want by
        # default to do the cleanup
        utils.ensure_dir(definitions.TEST_FORMULAS_DIR)
        utils.ensure_dir(definitions.TEST_SAMPLES_DIR)
        for config in os.listdir(definitions.BNN_TEST_CFG):
            parser = arg_parser.create_parser()
            args = parser.parse_args(['--results_dir', definitions.ROOT_BNN,
                                      'bnn', '--dataset', dataset, '--config',
                                      os.path.join(definitions.BNN_TEST_CFG,
                                                   config), 'encode'])
            print('[{}] {}'.format(self._testMethodName, args))
            nn = bnn_dataset.BNN(args, stats)
            nn.trained_models_dir = os.path.join(definitions.ROOT_BNN, nn.filename,
                                                 'train')
            nn.saved_model = os.path.join(nn.trained_models_dir, nn.filename +
                                          '.pt')
            nn.load_model(save=True)
            #output_filename = os.path.join(definitions.TEST_FORMULAS_DIR, args.encoder +
            #                       '-' + nn.filename + '-test.dimacs')
            #converter = mc.BNNConverter(nn.model_dir, output_filename)
            encoder = mc.BNNConverter('best')
            formula_fname = encoder.encode([nn])
            print('output {}'.format(formula_fname))
            #print('Encode {} and save CNF in {}'.format(os.path.basename(config), output_filename))
            #converter.encode()

    def test_all_bnn_configs(self):
        pool = Pool(processes=20)

        test_args = []
        for config in os.listdir(definitions.BNN_TEST_CFG):
            parser = arg_parser.create_parser()
            args = parser.parse_args(['--results_dir', definitions.ROOT_BNN,
                                      'bnn', '--dataset', dataset,'--config',
                                      os.path.join(definitions.BNN_TEST_CFG,
                                                   config), 'encode'])
            test_args.append((args, stats))

        res = pool.map(run_bnn_cfg, test_args)

        for r in res:
            self.assertEqual(r[1], 1.0)

if __name__ == '__main__':
    unittest.main()
