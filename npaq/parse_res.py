#!/usr/bin/env python

import csv
import os
import sys

def parse_row(row):
    fname = os.path.basename(row[0])
    # get rid of extension
    fname = os.path.splitext(fname)[0]
    # expecting encoder-dataset-input_size-name_nn-dp-label_id.dimacs
    fields = fname.split('-')
    nn_size = fields[2]
    name = fields[3]
    # get the dataset and label_id
    label_id = fields[-1]
    dataset = fields[1]

    return label_id, dataset, name, nn_size

def count_str_to_num(count_str_res):
    res_term1 = float(count_str_res.split('x')[0])
    res_term2 = 2**float(count_str_res.split('x')[1].split('^')[1])

    return res_term1 * res_term2


def parse_adj_res(csv_1_path, csv_2_path, out_path='.'):
    """
    Returns the ratio between dataset and adj_dataset from the csv result files.
    """
    if not os.path.exists(csv_1_path):
        print('No such file {}'.format(csv_1_path))
        exit(1)

    if not os.path.exists(csv_2_path):
        print('No such file {}'.format(csv_2_path))
        exit(1)

    res_dict = {}
    csvfile1 = open(csv_1_path, 'rb')
    csvfile2 = open(csv_2_path, 'rb')
    reader1 = csv.reader(csvfile1, delimiter=',')
    reader2 = csv.reader(csvfile2, delimiter=',')
    for row in reader1:
        label_id, dataset, name, _ = parse_row(row)
        res_dict[label_id] = [(dataset, row[1])]

    nn1_name = name
    for row in reader2:
        label_id, dataset, name, nn_size = parse_row(row)
        if not name == nn1_name:
            print('Different architecture {} - {}'.format(name, nn1_name))
        res_dict[label_id].append((dataset, row[1]))

    csvfile1.close()
    csvfile2.close()

    for label in res_dict:
        dataset1, res1 = res_dict[label][0]
        dataset2, res2 = res_dict[label][1]

        if res1 == 'timeout' or res2 == 'timeout':
            res_dict[label].append('timeout')
            continue
        res1 = count_str_to_num(res1)
        res2 = count_str_to_num(res2)

        final_res = res1 / res2
        res_dict[label].append(final_res)

    print(dataset1, dataset2)
    out_fname = dataset1 + '-' + dataset2 + '-' + nn_size + '-' + name + '.csv'
    print(out_fname)
    print(res_dict)

    out_fname = os.path.join(out_path, out_fname)
    with open(out_fname, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['NN', 'Label', dataset1, dataset2, 'd1/d2'])

        for label in res_dict:
            csvwriter.writerow([nn_size+'-'+name, label, res_dict[label][0][1],
                                res_dict[label][1][1], res_dict[label][2]])


def main():
    if len(sys.argv) < 2:
        print('Error: expecting 2 csv files as arguments. Exiting...')
        exit(1)

    parse_adj_res(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
