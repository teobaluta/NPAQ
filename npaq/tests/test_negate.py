#!/usr/bin/env python

from __future__ import print_function


import os
import unittest
import utils
import definitions
import negate
import subprocess
import shutil
import z3

TEST1_FILE='./tests/tests_input/formulas/test1.dimacs'
TEST1_LITS = 17
TEST1_CLAUSES = 19
TEST1_VARS = 5
TEST1_OUT = 1

class TestNegate(unittest.TestCase):

    def test_example_negate_self(self):
        if not os.path.exists(definitions.EXAMPLE_PATH):
            print('{} does not exist! Exiting...'.format(definitions.EXAMPLE_PATH))
            exit(1)

        subprocess.call([definitions.EXAMPLE_PATH, os.path.abspath(TEST1_FILE)])

        z3_fml, seen_lits = utils.dimacs2fml(TEST1_FILE)
        # print('[{}] z3 formula: {}'.format(self._testMethodName, z3_fml))
        print('[{}] #vars: {}'.format(self._testMethodName, seen_lits))
        self.assertEqual(len(seen_lits), 17)

        g = z3.Goal()
        neg_z3_fml = z3.Not(z3.And(z3_fml))
        # print('[{}] Negated formula: {}'.format(self._testMethodName, neg_z3_fml))
        g.add(neg_z3_fml)
        t = z3.Then('simplify', 'tseitin-cnf')
        neg_cnf = t(g)[0]
        #print('[{}] Negated CNF: {}'.format(self._testMethodName, neg_cnf))

        utils.translate2dimacs(neg_cnf, len(seen_lits), TEST1_FILE, append_to_file=True)
        num_vars, num_clauses = utils.parse_dimacs_header(TEST1_FILE)

        print('[{}] Formula and not formula num_vars {}, num_clauses \
              {}'.format(self._testMethodName, num_vars, num_clauses))

        fml, seen_lits = utils.dimacs2fml(TEST1_FILE)
        s = z3.Solver()
        s.add(fml)
        sol = s.check()
        self.assertEqual(sol, z3.unsat)

    def test_example_neg(self):
        if not os.path.exists(definitions.EXAMPLE_PATH):
            print('{} does not exist! Exiting...'.format(definitions.EXAMPLE_PATH))
            exit(1)

        print('[{}] generate test input file: {}'.format(self._testMethodName, TEST1_FILE))
        subprocess.call([definitions.EXAMPLE_PATH, os.path.abspath(TEST1_FILE)])
        out_and_cnf = './tests/tests_input/formulas/test1_not_test1.dimacs'
        negate.negate(TEST1_FILE, TEST1_FILE, TEST1_VARS, TEST1_OUT,
                      out_and_cnf)
        # TODO make this more testable
        # print('[{}] Writing to {}'.format(self._testMethodName, out_and_cnf))
        # # copy NN1 CNF to output file
        # shutil.copyfile(TEST1_FILE, out_and_cnf)

        # num_vars, num_clauses = utils.parse_dimacs_header(TEST1_FILE)
        # print('[{}] #vars: {}'.format(self._testMethodName, num_vars))
        # self.assertEqual(num_vars, 17)
        # self.assertEqual(num_clauses, 19)

        # reserved_vars = 5
        # z3_fml, seen_lits = utils.dimacs2fml(TEST1_FILE, reserved_vars,
                                             # num_vars, debug=True)
        # print('[{}] #vars: {}'.format(self._testMethodName, seen_lits))
        # self.assertEqual(len(seen_lits), 17)

        # z3_fml = z3.And(z3_fml)
        # print(z3_fml)

        # g = z3.Goal()
        # neg_fml = z3.Not(z3_fml)
        # t = z3.Then('simplify', 'tseitin-cnf')
        # g.add(neg_fml)
        # cnf_fml = t(g)[0]

        # print("CNF FML {}".format(cnf_fml))

        # utils.translate2dimacs(cnf_fml, num_vars + len(seen_lits) + 1 -
                               # reserved_vars, out_and_cnf, append_to_file=True)

        fml, seen_lits = utils.dimacs2fml(out_and_cnf)

        s = z3.Solver()
        s.add(fml)
        sol = s.check()
        print('[{}] test1 and not test1 = {}.'.format(self._testMethodName, sol))
        self.assertEqual(sol, z3.unsat)


    # def test_example_1_and_not_2(self):
        # TEST1_FILE = './test1.dimacs'
        # example2_output = './test2.dimacs'

        # if not os.path.exists(definitions.EXAMPLE_PATH):
            # print('{} does not exist! Exiting...'.format(EXAMPLE_PATH))
            # exit(1)

        # subprocess.check_call([definitions.EXAMPLE_PATH,
                               # os.path.abspath(TEST1_FILE)])


def main():
    unittest.main()

