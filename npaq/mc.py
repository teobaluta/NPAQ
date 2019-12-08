#!/usr/bin/env python

from __future__ import print_function

from abc import abstractmethod
import utils
import z3
import os
import subprocess
import logging
import time
#from definitions import BNN2CNF_PATH, FORMULAS_DIR
from definitions import BNN2CNF_PATH, FORMULAS_DIR, TEST_FORMULAS_DIR

from multiprocessing import Process

from enum import Enum

class PropType(Enum):
    MODEL = 1
    ROBUSTNESS = 2
    DISSIMILARITY = 3
    DP = 4
    TROJAN = 5
    FAIR = 6

logger = logging.getLogger(__name__)

class Converter(object):

    @abstractmethod
    def encode(self):
        pass

    def test_enc(self, samples, cnf_file):
        """Test the encoding using model samples.

        """
        # FIX THIS
        if not os.path.isfile(cnf_file):
            print('No encoding found!')
            exit(1)

        print('Loading CNF from {}'.format(cnf_file))
        # this returns a list of lists of ints
        cnf = utils.parse_dimacs(cnf_file)

        accuracy = 0.0
        cnf_assertions = []
        for clause in cnf:
            z3_clause = []
            # negated literals, e.g. -3, should be z3.Not(z3.Bool('3'))
            for lit in clause:
                if lit < 0:
                    z3_clause.append(z3.Not(z3.Bool(str(abs(lit)))))
                else:
                    z3_clause.append(z3.Bool(str(lit)))
            cnf_assertions.append(z3.Or(z3_clause))

        for (sample_in, sample_out) in samples:
            #print('Model sample/assumptions {} {}'.format(sample_in, sample_out))

            # XXX this is very shitty, have to create z3 objects
            # FIXME if the encoding is integer then the self.cnf is a z3 formula
            # already? -- need to check
            solver = z3.Solver()
            solver.add(cnf_assertions)
            #print('Sample in: {}; out {}'.format(sample_in, sample_out))
            solver.add(self.__bool_test_enc(sample_in, sample_out))
            sol = solver.check()
            if sol == z3.unsat:
                print('{} - Unsat for sample_in {}, sample_out \ {}'.format(cnf_file, sample_in, sample_out))
                continue
            m = solver.model()
#                 for i in range(1, 7):
                    # print("in x{} = {}".format(i, m[z3.Bool(str(i))]))

                # print("out x{} = {}".format(8, m[z3.Bool(str(26))]))
                # for i in range(14, 34):
                    # print('out x{} = {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(1, 5):
                    # print('in x{} = {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(5, 15):
                    # print("out x{} = {}".format(i, m[z3.Bool(str(i))]))

                # for i in range(52, 57):
                    # print('blk1 out {} = {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(57, 62):
                    # print('blk2 out {} = {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(27, 52):
                    # print('blk3 out {} = {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(108, 118):
                    # print('d vars {} {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(95, 103):
                    # print('blk1 out {} = {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(35, 125):
                    # print('d vars {} {}'.format(i, m[z3.Bool(str(i))]))

                # for i in range(36, 56):
                    # print("blk1out x{} = {}".format(i, m[z3.Bool(str(i))]))

                # # for i in range(56, 65):
                    # # print("d x{} = {}".format(i, m[z3.Bool(str(i))]))

                # for i in range(56, 146):
                    # print("x{} = {}".format(i, m[z3.Bool(str(i))]))

                #print("traversing model...")
                # for d in m.decls():
                    # print("{} = {}".format(d.name(), m[d]))
            if sol == z3.sat:
                accuracy += 1

        accuracy /= len(samples)
        # print('[{}] encoding accuracy: {}'.format(self.model.name, accuracy))
        #logger.debug('Encoding accuracy: {}'.format(accuracy))

        return accuracy

    def __bool_test_enc(self, sample_in, sample_out, bin_vector_out=True):
        assertions = []
        for idx, sample_in_var in enumerate(sample_in):
            in_var = z3.Bool(str(idx + 1))
            if sample_in_var == 1:
                assertions.append(in_var == True)
            else:
                assertions.append(in_var == False)

        if bin_vector_out:
            #print('Sample out: {}'.format(sample_out))
            for idx, sample_out_var in enumerate(sample_out):
                out_var = z3.Bool(str(len(sample_in) + idx + 1))
                if sample_out_var == 1:
                    assertions.append(out_var == True)
                else:
                    assertions.append(out_var == False)
        else:
            s = z3.Solver()
            s.check()
            m = s.model()

            # FIXME this should be used only with the method _enc_integer_out
            out_var_idx, out_var = next(self.out_vars.iteritems())
            sample_out_val = z3.BitVecVal(sample_out[0], out_var.precision)
            bits = [z3.Extract(i, i, sample_out_val) for i in \
                    range(sample_out_val.size())]
            eval_bits = [m.evaluate(bits[i]) for i in range(out_var.precision)]

            logger.debug('sample out {}; bits {}'.format(sample_out_val, eval_bits))

            for bit, bool_var in out_var.bitmap.iteritems():
                if eval_bits[bit] == 1:
                    assertions.append(bool_var == True)
                else:
                    assertions.append(bool_var == False)

        return assertions

class BNNConverter(Converter):
    def __init__(self, enc_strategy='best', debug=False):
        if not enc_strategy in ['card', 'best', 'bdd']:
            print('Unknown enc_strategy %s' % enc_strategy)
            exit(1)

        self.enc_strategy = '--' + enc_strategy
        if debug:
            self.debug = '--debug'

    def _nn_formula_name(self, name):
        #utils.ensure_dir(FORMULAS_DIR)
        utils.ensure_dir(TEST_FORMULAS_DIR)
        return os.path.join(TEST_FORMULAS_DIR, self.enc_strategy[2:] + '-' +
                            name + '.dimacs')

    def _diff_formula_name(self, name1, name2):
        utils.ensure_dir(FORMULAS_DIR)

        filename = '{}-{}-diff-{}.dimacs'.format(self.enc_strategy[2:], name1,
                                                 name2)
        fname = os.path.join(FORMULAS_DIR, filename)

        return fname

    def _diff_w_ct_formula_name(self, name1, name2):
        utils.ensure_dir(FORMULAS_DIR)

        filename = '{}-{}-diff-w_ct-{}.dimacs'.format(self.enc_strategy[2:], name1,
                                                 name2)
        fname = os.path.join(FORMULAS_DIR, filename)

        return fname



    def _robust_formula_name(self, name, perturb_k, perturb_num):
        utils.ensure_dir(FORMULAS_DIR)

        filename = '{}-{}-robustness-perturb_{}-id_{}.dimacs'.format(
            self.enc_strategy[2:], name, perturb_k, perturb_num)
        fname = os.path.join(FORMULAS_DIR, filename)

        return fname

    def _dp_formula_name(self, name, label):
        utils.ensure_dir(FORMULAS_DIR)

        filename = '{}-{}-dp-label_{}.dimacs'.format(self.enc_strategy[2:],
                                                     name, label)
        fname = os.path.join(FORMULAS_DIR, filename)

        return fname

    def _trojan_formula_name(self, name, label):
        utils.ensure_dir(FORMULAS_DIR)

        filename = '{}-{}-trojan-label_{}.dimacs'.format(self.enc_strategy[2:],
                                                     name, label)
        fname = os.path.join(FORMULAS_DIR, filename)

        return fname

    def _fair_formula_name(self, name, constraints_fname):
        utils.ensure_dir(FORMULAS_DIR)

        constraints_fname = os.path.splitext(os.path.basename(constraints_fname))[0]
        filename = '{}-{}-{}-fair.dimacs'.format(self.enc_strategy[2:], name,
                                                 constraints_fname)
        fname = os.path.join(FORMULAS_DIR, filename)

        return fname

    def encode(self, neural_nets, args=[], prop_type=PropType.MODEL):
        call_args = [BNN2CNF_PATH]

        if not isinstance(neural_nets, list):
            print('encode expects a list of models')
            exit(1)

        if prop_type == PropType.MODEL:
            nn = neural_nets[0]
            formula_fname = self._nn_formula_name(nn.filename)

            call_args += [nn.model_dir, formula_fname, self.enc_strategy]

        elif prop_type == PropType.DISSIMILARITY:
            if len(neural_nets) < 2:
                logger.debug('[Error] expected 2 models for %s' % PropType.DISSIMILARITY)
                exit(1)

            nn1 = neural_nets[0]
            nn2 = neural_nets[1]
            if len(args)  == 0:
                formula_fname = self._diff_formula_name(nn1.filename, nn2.filename)

                call_args += [os.path.abspath(nn1.model_dir), formula_fname, 'diff',
                              os.path.abspath(nn2.model_dir)]

            else:
                constraints_fname = args[0]
                formula_fname = self._diff_w_ct_formula_name(nn1.filename, nn2.filename)

                call_args += [os.path.abspath(nn1.model_dir), formula_fname, 'diff',
                              os.path.abspath(nn2.model_dir), constraints_fname]

        elif prop_type == PropType.ROBUSTNESS:
            nn = neural_nets[0]
            if len(args) < 2:
                print('[Error] expecting perturbation and concrete input filename')
                exit(1)

            perturb = args[0]
            concrete_ip_fname = args[1]

            equal = args[2]
            if len(args) == 3:
                idx = 0
            else:
                idx = args[3]


            formula_fname = self._robust_formula_name(nn.filename, perturb, idx)
            if equal == True:
                call_args += [os.path.abspath(nn.model_dir), formula_fname,
                              'robust', perturb,
                              os.path.abspath(concrete_ip_fname), '--equal']
            else:
                call_args += [os.path.abspath(nn.model_dir), formula_fname,
                              'robust', perturb,
                              os.path.abspath(concrete_ip_fname)]
        elif prop_type == PropType.DP:
            nn = neural_nets[0]

            if len(args) < 1:
                print('[Error] expecting first arg to be label')
                exit(1)

            label = args[0]
            formula_fname = self._dp_formula_name(nn.filename, label)
            # at the moment the property that we settled on encoding is 
            # for all x, bnn(x) == label => Pr[bnn(x) == l]
            call_args += [os.path.abspath(nn.model_dir), formula_fname, 'label',
                          str(label)]
        elif prop_type == PropType.TROJAN:
            nn = neural_nets[0]

            if len(args) < 1:
                print('[Error] expecting first arg to be label')
                exit(1)

            label = args[0]
            constraints_fname = args[1]
            formula_fname = self._trojan_formula_name(nn.filename, label)
            # at the moment the property that we settled on encoding is 
            # for all x, bnn(x) == label => Pr[bnn(x) == l]
            call_args += [os.path.abspath(nn.model_dir), formula_fname, 'label',
                          str(label), constraints_fname]
        elif prop_type == PropType.FAIR:
            nn = neural_nets[0]

            if len(args) < 1:
                print('[Error] expecting first arg to be label')
                exit(1)

            constraints_fname = args[0]
            dataset_ct_fname = args[1]
            formula_fname = self._fair_formula_name(nn.filename,
                                                    constraints_fname)
            call_args += [os.path.abspath(nn.model_dir), formula_fname, 'fair',
                          os.path.abspath(constraints_fname),
                          os.path.abspath(dataset_ct_fname)]

        if self.enc_strategy[2:] == 'card':
            call_args += [self.enc_strategy]

        logger.debug('call bnn2cnf with {}'.format(call_args))
        start_time = time.time()
        process = subprocess.Popen(call_args, stdout=subprocess.PIPE)
        stdout = process.communicate()[0]
        logger.debug(stdout)
        logger.debug('Encoding {} formula took {} seconds'.format(formula_fname,
                                                                  time.time() - start_time))
        ret_code = process.returncode
        if ret_code:
            logger.debug('%s returned with %s code' % (call_args, ret_code))
            exit(1)


        return formula_fname
