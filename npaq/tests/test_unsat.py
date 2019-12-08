#!/usr/bin/env python

from __future__ import print_function

import unittest
import os
import mc
import arg_parser
import models
import stats
import pickle
import definitions
import utils
import z3
import subprocess

from multiprocessing import Pool

stats = stats.RecordStats('tests_unsat_stats.txt')

def run_unsat_call_cpp(v):
    args = v[0]
    stats = v[1]
    nn = models.bnn.BNNmnist(args, stats)

    call_args = [definitions.BNN2CNF_PATH, os.path.abspath(nn.model_dir),
                 os.path.abspath(nn.enc_out_dir), 'diff',
                 os.path.abspath(nn.model_dir)]
    process = subprocess.Popen(call_args, stdout=subprocess.PIPE)
    stdout = process.communicate()[0]

    z3_fml, _ = utils.dimacs2fml(os.path.join(nn.enc_out_dir,
                                              definitions.DIFF_OUT_FILE))

    solver = z3.Solver()
    solver.add(z3_fml)

    return solver.check()


class TestUnsat(unittest.TestCase):

    def test_unsat_diff_cpp(self):
        pool = Pool(processes=20)

        test_args = []
        for config in os.listdir(definitions.BNN_TEST_CFG):
            parser = arg_parser.create_parser()
            # these are some dummy args just so we can create the BNNmnist and
            # initialize the correct paths of model_dir and enc_out_dir
            args = parser.parse_args(['--results_dir', definitions.ROOT_BNN,
                                      'bnn-mnist', '--config',
                                      os.path.join(definitions.BNN_TEST_CFG,
                                                   config), 'test_enc'])
            test_args.append((args, stats))

        res = pool.map(run_unsat_call_cpp, test_args)

        for sol in res:
            if sol == z3.unsat:
                print('[{}] PASS {}'.format(self._testMethodName, args.config))
            self.assertEqual(sol, z3.unsat)


