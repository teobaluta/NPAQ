import os
import csv

def read_csv(csv_file):
    content = []
    with open(csv_file) as f:
        csv_reader = csv.reader(f,  delimiter=',')
        for row in csv_reader:
            content.append(row)
    return content

def record_log(log_str, log_path):
    with open(log_path, 'a+') as f:
        f.write(log_str+'\n')
    print(log_str)

def ensure_dir(dir):
    if os.path.exists(dir):
        pass
    else:
        os.mkdir(dir)

def write_bin(sample_fname, bin_sample):
    with open(sample_fname, 'wb') as f:
        f.write(bytearray(list(bin_sample)))