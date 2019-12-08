#!/usr/bin/env python

from __future__ import print_function

import random

from definitions import BNN2CNF_PATH
import stats
import mc
import os
import subprocess
import utils

stats = stats.RecordStats('stats.txt')

class AdversarialBNN(object):
    # XXX change the hardcoded version of this
    # would be better to have a json description of the network but now will
    # just parse based on the model directory under args.save_dir
    def __init__(self, args):
        print(args.save_dir)
        print(args.k_perturb)

        # this one could also read from the input
        self.input_size = 100
        self.k_perturb = args.k_perturb
        self.path = args.save_dir
        self.model_dir = os.path.join(self.path, 'model')
        if not os.path.exists(self.model_dir):
            print('Expecting model dir {} where params are.'.format(self.model_dir))
            exit(1)

        self.num_internal_blocks = 0
        for folder_name in os.listdir(self.model_dir):
            if folder_name.startswith("blk"):
                self.num_internal_blocks += 1

        print('num internal blocks found {}'.format(self.num_internal_blocks))

        self.enc_out_dir = os.path.join(self.path, 'encoding')
        self.cnf_file = os.path.join(self.enc_out_dir, 'cnf.dimacs')
        utils.ensure_dir(self.enc_out_dir)
        super(AdversarialBNN, self).__init__(self.num_internal_blocks,
                                             self.path)
        self.args = args

    def encode(self):
        """
        The input variables are assumed to start from 1 to first block input size
        """
        random.seed(17)

        idx = []
        while len(idx) < self.k_perturb:
            rand_x = random.randint(1, self.input_size)
            idx.append(rand_x)

        print('k-flipped bits {}'.format(idx))
        adv_perturb_path = os.path.join(self.args.save_dir, 'adv-perturb-' +
                                        str(self.k_perturb))

        with open(adv_perturb_path, 'w') as f:
            for i in idx:
                f.write('%s ' % i)

        if not os.path.exists(BNN2CNF_PATH):
            print('{} binary does not exist! Existing...'.format(BNN2CNF_PATH))
            exit(1)

        path = os.path.abspath(self.enc_out_dir)
        call_args = [BNN2CNF_PATH, os.path.abspath(self.model_dir), path,
                     adv_perturb_path]
        subprocess.check_call(call_args)

