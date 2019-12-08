#!/usr/bin/env python

from __future__ import print_function

import os
import ConfigParser
import sys
import z3
import logging
import shutil
import time
import definitions
import pickle
import math
import imageio
import datetime
import gzip

logger = logging.getLogger(__name__)


def ensure_dir(path):
    logger.debug(path)
    if path and (not os.path.isdir(path)):
        os.makedirs(path)

def query(question, default=""):
    print('{} [{}]: '.format(question, default), end='')
    choice = raw_input()
    if choice == '':
        choice = default
    return choice


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")


def read_config(config_file):
    configs = {}
    config = ConfigParser.SafeConfigParser()
    config.read([config_file])
    for name in config.options('strings'):
        configs[name] = config.get('strings', name)

    for name in config.options('ints'):
        configs[name] = config.getint('ints', name)

    for name in config.options('booleans'):
        configs[name] = config.getboolean('booleans', name)

    register_set = config.get('regs', 'regs_set')
    register_set = [ int(x) for x in register_set.split(',')]
    configs['regs_set'] = register_set
    print('Register set {}'.format(configs['regs_set']))

    return configs


def parse_model_path(config, activation='sigmoid'):
    if 'base_path' not in config or 'instr' not in config:
        return None
    return os.path.join(config['base_path'], 'models', config['instr'],
                        activation, 'model')


def parse_data_path(config):
    if 'data_path' in config:
        return config['data_path']

    if 'instr' not in config:
        return None

    return os.path.join(config['base_path'], 'data', config['instr'])


"""
DIMACS format parsing
"""

def parse_dimacs_header(dimacs_file):
    """Parses the header of a dimacs file.

    Return (number of variables,  number of clauses).
    """
    header = ""
    with open(dimacs_file, 'r') as f:
        header = f.readline()

    if not header.startswith('p cnf '):
        print('{} not in DIMACS format!'.format(dimacs_file))
        return (None, None)

    header_tokens = header.split()
    return (int(header_tokens[2]), int(header_tokens[3]))


def parse_dimacs(dimacs_file):

    clauses = []
    with open(dimacs_file, 'r') as f:
        header = f.readline()
        if not header.startswith('p cnf '):
            print('{} not in DIMACS format!'.format(dimacs_file))
            return (None, None)
        header_tokens = header.split()
        num_clauses = int(header_tokens[3])

        for line in f:
            if line.startswith('c ind '):
                print('{} contains c ind. Skip'.format(dimacs_file))
            else:
                try:
                    clauses.append([int(x) for x in line.split(' ')[:-1]])
                except:
                    print(line)
                    print(line.split(' '))

        assert(len(clauses) == num_clauses)
        return clauses


def translate2(lit, seen_literals):

    or_clause = ''
    if z3.is_not(lit):
        or_clause = '-'
        lit = lit.children()[0]

    if str(lit)[0] == 'k':
        or_clause += str(lit)[2:] + ' '

        if not lit in seen_literals:
            seen_literals[lit] = str(lit)
    else:
        or_clause += str(lit)[1:] + ' '

    return or_clause


def formula2dimacs(z3_formula, output_filename, append_to_file=False,
                   verbose=False):
    seen_literals = {}
    number_of_clauses = 0

    clauses = []
    if append_to_file:
        if not os.path.isfile(output_filename):
            print('{} does not exist! Exiting...')
            exit(1)

    cnf_clauses = []
    # Z3 expressions can be applications, quantifiers and bounded/free variables
    # FIXME
    for expr in z3_formula:
        if z3.is_quantifier(expr):
            print('Expecting {} to be OR, got quantifier in CNF!'.format(expr))
            return

        #print(expr)
        if z3.is_or(expr):
            or_clause = ''.join([translate2(lit, seen_literals) for lit in
                                 expr.children()])
            cnf_clauses.append(or_clause)
        else:
            lit = translate2(expr, seen_literals)
            cnf_clauses.append(lit)

    if not append_to_file:
        with open(output_filename, 'w') as f:
            f.write('p cnf {} {}\n'.format(len(seen_literals),
                                           len(cnf_clauses)))
            for clause in cnf_clauses:
                f.write('{}0\n'.format(clause))
    else:
        (num_vars, num_clauses) = parse_dimacs_header(output_filename)
        with open(output_filename, 'r') as from_file:
            tmp_path = output_filename + ".tmp"
            with open(tmp_path, 'w') as tmp_file:
                tmp_file.write('p cnf {} {}\n'.format(num_vars +
                                                      len(seen_literals),
                                                      num_clauses +
                                                      len(cnf_clauses)))
                line = from_file.readline()
                for line in from_file.readlines():
                    tmp_file.write(line)

                for clause in cnf_clauses:
                    tmp_file.write('{}0\n'.format(clause))

            os.rename(tmp_path, output_filename)

        # XXX why doesn't this work??
        # from_file = open(output_filename, 'r')
        # (num_vars, num_clauses) = parse_dimacs_header(output_filename)
        # line = from_file.readline()
        # print('Prev header {}'.format(line))

        # to_file = open(output_filename, 'wb')
        # to_file.write('p cnf {} {}\n'.format(num_vars + len(seen_literals),
                                             # num_clauses + len(cnf_clauses)))
        # shutil.copyfileobj(from_file, to_file)
        # from_file.close()
        # to_file.close()

def translate(start_var, lit, seen_literals):

    or_clause = ''
    if z3.is_not(lit):
        or_clause = '-'
        lit = lit.children()[0]

    if str(lit)[0] == 'k':
        or_clause += str(int(str(lit)[2:]) + start_var) + ' '
        if not lit in seen_literals:
            seen_literals[str(lit)] = int(str(lit)[2:]) + start_var

    elif str(lit)[0] == 'x':
        or_clause += str(lit)[1:]  + ' '
        if not lit in seen_literals:
            seen_literals[str(lit)] = int(str(lit)[1:])
    else:
        if not lit in seen_literals:
            seen_literals[str(lit)] = int(str(lit))
        or_clause += str(lit) + ' '

    return or_clause


def count_header(num_indep_vars):
    i = 1
    strs = []
    while i <= num_indep_vars:
        line = 'c ind'
        for j in range(0, 9):
            if i + j > num_indep_vars:
                break
            line += ' {}'.format(str(i + j))
        i += 9
        line += ' 0'
        strs.append(line)

    return '\n'.join(strs)


def translate2dimacs(z3_formula, start_var, output_filename,
                     append_to_file=False, overlap_vars=0, indep_set=0,
                     verbose=False):
    """Write a z3 formula to a DIMACS output format.

    Parameters:
    - start_var: Offset the auxiliary variables by start_var
    """
    seen_literals = {}

    cnf_clauses = []
    start = time.time()
    # Z3 expressions can be applications, quantifiers and bounded/free variables
    for expr in z3_formula:
        if z3.is_quantifier(expr):
            print('Expecting {} to be OR, got quantifier in CNF!'.format(expr))
            return
        if z3.is_or(expr):
            # for lit in expr.children():
               # or_clause += translate(start_var, lit, seen_literals)
            or_clause = ''.join([translate(start_var, lit, seen_literals) for
                                 lit in expr.children()])
            cnf_clauses.append(or_clause)
        else:
            lit = translate(start_var, expr, seen_literals)
            if verbose:
                print('{} not OR. translated to {}'.format(expr, lit))
            cnf_clauses.append(lit)

    number_of_clauses = len(cnf_clauses)
    end = time.time()
    if verbose:
        print('SEEN LITERALS {}'.format(seen_literals))

    print('Iterating took {} sec'.format(end - start))

    if not append_to_file:
        with open(output_filename, 'w') as f:
            f.write('p cnf {} {}\n'.format(len(seen_literals) + start_var,
                                           number_of_clauses))
            if indep_set > 0:
                f.write('{}\n'.format(count_header(indep_set)))
            curr = 0
            prev = 0
            for curr, clause in enumerate(cnf_clauses):
                curr += 1
                # flush every 32K elements -- this 32k is an approx.
                if curr % 99991 == 0:
                    f.write('0\n'.join(cnf_clauses[prev:curr]))
                    f.write('0\n')
                    prev = curr
            #f.write('0\n'.join(cnf_clauses))
            f.write('0\n'.join(cnf_clauses[prev:curr]))
            f.write('0\n')
    else:
        (num_vars, num_clauses) = parse_dimacs_header(output_filename)
        with open(output_filename, 'r') as from_file:
            tmp_path = output_filename + '.tmp'
            with open(tmp_path, 'w') as tmp_file:
                number_of_clauses += num_clauses
                tmp_file.write('p cnf {} {}\n'.format(num_vars + start_var +
                                                      len(seen_literals) -
                                                      overlap_vars,
                                                      number_of_clauses))

                if indep_set > 0:
                    tmp_file.write('{}\n'.format(count_header(indep_set)))

                line = from_file.readline()
                start = time.time()
                shutil.copyfileobj(from_file, tmp_file)
                end = time.time()
                logger.debug('shutil.copyfileobj took {} sec'.format(end -
                                                                     start))
                curr = 0
                prev = 0
                if len(cnf_clauses) < 99991:
                    tmp_file.write('0\n'.join(cnf_clauses))
                    tmp_file.write('0\n')
                else:
                    for curr, clause in enumerate(cnf_clauses):
                        curr += 1
                        # flush every 100K elements -- this 32k is an approx.
                        if curr % 99991 == 0:
                            tmp_file.write('0\n'.join(cnf_clauses[prev:curr]))
                            tmp_file.write('0\n')
                            prev = curr
                    tmp_file.write('0\n'.join(cnf_clauses[prev:curr]))
                    tmp_file.write('0\n')

                #tmp_file.write('0\n'.join(cnf_clauses))
            os.rename(tmp_path, output_filename)

    return len(seen_literals), number_of_clauses

def dimacs2fml(cnf_dimacs, reserved_vars=0, var_offset=0, debug=False):
    """Return a z3 CNF formula from DIMACS file.

    Parameters:
    - reserved_vars: will translate any literals after reserved_vars literals.
    Hence, if reserved_vars=0 all of the literals will be offset wil var_offset
    - var_offset: literals will be offset by (var_offset - reserved_vars)

    """

    cnf = parse_dimacs(cnf_dimacs)
    big_and = []
    seen_literals = {}
    for clause in cnf:
        z3_clause = []
        # negated literals, e.g. -3, should be z3.Not(z3.Bool('3'))
        for lit in clause:
            if lit < 0:
                # if there's any specified reserved_vars offset the auxiliary
                # variables
                lit = abs(lit)
                if reserved_vars > 0:
                    if lit > reserved_vars:
                        #print('neg lit {} offset += {}'.format(lit, var_offset))
                        seen_literals[lit] = lit - reserved_vars + var_offset
                        lit = lit - reserved_vars + var_offset
                    else:
                        seen_literals[lit] = lit
                else:
                    seen_literals[lit] = lit
                z3_clause.append(z3.Not(z3.Bool(str(lit))))
            else:
                if reserved_vars > 0:
                    if lit > reserved_vars:
                        #print('lit {} offset += {}'.format(lit, var_offset))
                        seen_literals[lit] = lit - reserved_vars + var_offset
                        lit = lit - reserved_vars + var_offset
                    else:
                        seen_literals[lit] = lit
                else:
                    seen_literals[lit] = lit
                z3_clause.append(z3.Bool(str(lit)))

        big_and.append(z3.Or(z3_clause))

    return big_and, seen_literals


def check_folder(folder_path):
    if os.path.exists(folder_path):
        pass
    else:
        raise Exception('ERROR: Folder does not exist! => %s' % folder_path)

def write_sample(sample_fname, bin_sample):

    with open(sample_fname, 'wb') as f:
        f.write(bytearray(list(bin_sample)))

def get_output_fnames(formula_fname):
    ensure_dir(definitions.COUNT_OUT_DIR)

    formula_basefname = os.path.basename(formula_fname)
    output_scalmc_fname = os.path.join(definitions.COUNT_OUT_DIR,
                                       formula_basefname + '.out')

    return output_scalmc_fname

def set_logger_prop(dataset, prop_name, filename):
    logs_dir = os.path.join(definitions.RESULTS_DIR, 'logs')
    ensure_dir(logs_dir)
    logs_dir = os.path.join(logs_dir, '%s_%s' % (dataset, prop_name))
    ensure_dir(logs_dir)
    print('Log Dir -> %s' % logs_dir)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s \
                        [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=os.path.join(logs_dir, filename +
                                              '-{}.log'.format(datetime.datetime.now())),
                        level=logging.DEBUG)


def write_pkl(info, file_path):
    with open(file_path, 'w') as f:
        pickle.dump(info, f)

def save_img(bin_vec, img_path):
    img_size =  int(math.sqrt(len(bin_vec)))

    img = []
    for i in range(img_size):
        img.append([])
        for j in range(img_size):
            img[i].append(bin_vec[img_size * i + j])

    imageio.imwrite(img_path, img)

def save_pil(img, img_path):
    imageio.imwrite(img_path, img)

def dataset_split_names(dataset):
    """ Convention of naming dataset splits
    """
    return dataset + '_1', dataset + '_2'

def adj_dataset(dataset, seed):
    return '{}_dp-seed_{}'.format(dataset, seed)


def gzip_formula(src_file, dst_file):
    print(src_file, dst_file)
    with open(src_file, 'rb') as f_in, gzip.open(dst_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

