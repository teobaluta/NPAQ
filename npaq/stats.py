#!/usr/bin/env python

from __future__ import print_function

import json
import utils
import csv
from beautifultable import BeautifulTable

class RecordStats(object):
    def __init__(self, stats_file):
        self.stats = {}
        self.stats_file = stats_file

    def record_descr(self, nn_name, descr):
        """
        Describe the architecture of the network. Each neural network implements
        their own representation of the object descr.
        """
        self.stats[nn_name] = {'descr': descr}

    def record_test(self, nn_name, test_size, test_loss, test_accuracy):
        self.stats[nn_name] = {
            'test': {
                'test_size': test_size,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
        }

    def record_train(self, nn_name, best_model_id, train_size, train_loss,
                     train_accuracy):
        self.stats[nn_name] = {
            'train': {
                'best_model_id': best_model_id,
                'epoch': epoch,
                'train_size': train_size,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy
            }
        }

    def pp(self):
        """ Pretty-print of the stats.
        """

        table = BeautifulTable()

        table.column_headers = ['Neural Net', 'Accuracy (%)']
        for nn, nn_info in self.stats.iteritems():
            table.append_row([nn, nn_info['test']['test_accuracy']])


    def dump(self):
        print('Write stats to file {}'.format(self.stats_file))
        print(self.stats)
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f)


    def to_csv(self):
        csvf = open('stats.csv', 'wb')
        csvwriter = csv.writer(csvf, delimiter=',')
        for nn, nn_stat in self.stats.iteritems():
            csvwriter.writerow([nn, nn_stat['test']['test_accuracy'],
                                nn_stat['test']['test_loss']])
