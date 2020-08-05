#!/usr/bin/env python

from __future__ import print_function

import os
import subprocess32
import definitions
import utils
import logging

import threading


"""
Scripts for invoking with certain arguments the counter.
"""

logger = logging.getLogger(__name__)

def output_reader(process, output_scalmc_fname):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        with open(output_scalmc_fname, 'a') as f:
            f.write(line)
        logger.debug(line.rstrip())

def realtime_scalmc(process, timeout, output_scalmc_fname):
    t = threading.Thread(target=output_reader, args=(process,
                                                     output_scalmc_fname))
    logger.debug('Starting thread {}'.format(t))
    t.start()

    if os.path.exists(output_scalmc_fname):
        os.remove(output_scalmc_fname)

    try:
        process.wait(timeout=timeout)
        logger.debug('subprocess terminated with {}'.format(process.returncode))
        res = 'finished'
    except subprocess32.TimeoutExpired:
        process.kill()
        out, errs = process.communicate()
        with open(output_scalmc_fname, 'a') as f:
            f.write(out)
            f.write('npaq-timeout\n')
        logger.debug('subprocess terminated with {}'.format(process.returncode))
        res = 'timeout'

    t.join()
    return res

def invoke_scalmc(formula_fname, proj_size, epsilon=0.8, delta=0.2,
                  timeout=1000, realtime=True, num_samples=10):
    """
    """
    start_iter = proj_size - 20

    # Depending on the scalmc version can ask for satisfying assingments and save these in a file
    # called 'samples_fname'
    output_scalmc_fname, samples_fname = utils.get_output_fnames(formula_fname)

    call_args = [definitions.COUNTER, '-s', '1', formula_fname]

    print('Started counting over {}'.format(formula_fname))
    logger.debug('Started counting over {}'.format(formula_fname))
    process = subprocess32.Popen(call_args, stdout=subprocess32.PIPE)

    if realtime:
        res = realtime_scalmc(process, timeout, output_scalmc_fname)
        if 'timeout' in res:
            return (formula_fname, res)

    else:
        try:
            # if not realtime buffers the stdout and logs after the process finishes
            # otherwise, will print out line by line the output of scalmc
            stdout = process.communicate(timeout=timeout)[0]
            logger.debug(stdout)
            with open(output_scalmc_fname, 'w') as f:
                f.write(stdout)
        except subprocess32.TimeoutExpired:
            process.kill()
            out, errs = process.communicate()
            logger.debug('scalmc TIMEOUT out: {}'.format(out))
            logger.debug('scalmc TIMEOUT err: {}'.format(errs))
            with open(output_scalmc_fname, 'w') as f:
                f.write('npaq-timeout\n')
                f.write(out)
            return (formula_fname, 'timeout')

    logger.debug('scalmc output in {}'.format(output_scalmc_fname))

    # check whether unsatisfiable
    logger.debug('Check if {} is unsat.'.format(formula_fname))
    with open(output_scalmc_fname, 'r') as f:
        output = f.readlines()
        unsat = output[-1]
        if 'unsatisfiable' in unsat:
            return (formula_fname, 'unsat')
        for line in output:
            if 'Number of solutions' in line:
                return (formula_fname, line.split(':')[1][1:][:-1])
