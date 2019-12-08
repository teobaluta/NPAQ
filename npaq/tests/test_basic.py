#!/usr/bin/env python

from __future__ import print_function

import unittest
import mc
import mc_vars

from collections import OrderedDict

class BNNTest(unittest.TestCase):

    def test_blk1(self):
        lin_1 = mc.LinearLayer([[1], [-1]],
                               [-0.5])

        bn_1 = mc.BatchNormLayer([0.12],
                                 [0.1],
                                 [-0.1],
                                 [2])

        blk1 = mc.BNNBlock('testBlk1', lin_1, bn_1)

        in_vars = OrderedDict()

        for i in range(blk1.in_size):
            x = mc_vars.Var.factory('bool', i + 1, 1)
            in_vars[x.idx] = x

        blk1.init_vars(in_vars, 'bool', 1)
        #print('{} in_vars {}'.format(blk1.name, blk1.in_vars))
        #print('{} out_vars {}'.format(blk1.name, blk1.out_vars))

        self.assertEqual(blk1.in_vars, OrderedDict([(1, in_vars[1]),
                                                    (2, in_vars[2])]))
        self.assertTrue(blk1.out_vars[3])

        #blk1.encode(cnf=True)
        #print('{} constr: {}'.format(blk1.name, blk1.enc))
        #print('{} CNF: {}'.format(blk1.name, blk1.cnf))


    def test_out(self):
        lin_layer = mc.LinearLayer([[1, -1], [-1, 1]],
                                   [-0.5, 0.2])
        out = mc.BNNOutBlock(lin_layer)

        in_vars = OrderedDict()

        for i in range(out.in_size):
            x = mc_vars.Var.factory('bool', i + 1, 1)
            in_vars[x.idx] = x

        out.init_vars(in_vars, 'bool', 1)
        print('{} in_vars {}'.format(out.name, out.in_vars))
        print('{} out_vars {}'.format(out.name, out.out_vars))
        print('{} d_vars {}'.format(out.name, out.d_vars))

        self.assertEqual(out.in_vars,  OrderedDict([(1, in_vars[1]),
                                                    (2, in_vars[2])]))
        self.assertTrue(out.d_vars[3])
        self.assertTrue(out.d_vars[4])
        self.assertTrue(out.d_vars[5])
        self.assertTrue(out.d_vars[6])

        #out.encode(cnf=True)
        #print('{} constr: {}'.format(out.name, out.enc))
        #print('{} CNF: {}'.format(out.name, out.cnf))


if __name__ == '__main__':
    unittest.main()
