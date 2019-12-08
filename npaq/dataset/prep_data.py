import os
import sys
import utils
import numpy as np
import definitions
import multiprocessing
from multiprocessing import Pool

LOG_PATH = ''
PROCESS_NUM = multiprocessing.cpu_count() - 2

def split_by_class(data, labels):
    '''

    :param data: numpy.array
    :param labels: numpy.array
    :return:
    '''
    sample_num = data.shape[0]
    split_info = {}
    for id in range(sample_num):
        if labels[id] in split_info:
            split_info[labels[id]].append(data[id])
        else:
            split_info[labels[id]] = [data[id]]
    return split_info

def write_bin_files(folder_path, images, tag=''):
    sample_num = len(images)
    pool = Pool(PROCESS_NUM)
    for id in range(sample_num):
        path = os.path.join(folder_path, '%s%d.bin' % (tag, id))
        # utils.write_bin(path, images[id])
        pool.apply_async(
            utils.write_bin,
            args = (path, images[id])
        )
    pool.close()
    pool.join()

def rebalance_dataset(split_info, sample_size):
    data = []
    labels = []
    for class_id in split_info:
        origin_size = len(split_info[class_id])
        print('Class %d: %d samples' % (class_id, origin_size))
        if origin_size > sample_size:
            temp = np.asarray(split_info[class_id])
            shuffle_list = np.arange(temp.shape[0])
            np.random.shuffle(shuffle_list)
            if len(data) == 0:
                data = temp[shuffle_list[:sample_size]]
                labels = np.asarray([class_id] * sample_size)
            else:
                data = np.concatenate((data, temp[shuffle_list[:sample_size]]), axis=0)
                labels = np.concatenate((labels, np.asarray([class_id] * sample_size)), axis = 0)
        else:
            if len(data) == 0:
                data = np.asarray(split_info[class_id])
                labels = np.asarray([class_id] * data.shape[0])
            else:
                data = np.concatenate((data, np.asarray(split_info[class_id])), axis=0)
                labels = np.concatenate((labels, np.asarray([class_id] * data.shape[0])), axis = 0)
    print('Final Data Size: %d' % data.shape[0])
    return data, labels

def real2bin(real_value, bit_num):
    temp = bin(real_value)[2:]
    temp = [int(i) for i in temp]
    temp = [0] * (bit_num-len(temp)) + temp
    return temp

def read_txt(file_path):
    with open(file_path) as f:
        content = f.readlines()
    return content

def process_uci_adult_dataset(content):
    content = [temp.split() for temp in content]
    content2 = []
    for item in content:
        if len(item) == 15:
            item2 = []
            for temp in item:
                if temp[-1] == ',':
                    item2.append(temp[:-1])
                elif temp[-1] == '.':
                    item2.append(temp[:-1])
                else:
                    item2.append(temp)
            content2.append(item2)
    content = np.asarray(content2)
    # clean dataset
    index_list = np.asarray(list(set(range(content.shape[0])) - \
                                 set(np.where(content == '?')[0]))).astype(np.int)
    content = content[index_list]
    labels = content[:, -1].astype(np.str)
    new_labels = np.asarray([-1] * labels.shape[0])
    index_list = np.where(labels == '<=50K')[0]
    new_labels[index_list] = 0
    index_list = np.where(labels == '>50K')[0]
    new_labels[index_list] = 1
    labels = new_labels.astype(np.int)

    feature_info = {
        0: {'type': 'float', 'start': 10, 'range': 10.0, 'bit_num': 3},  # age
        1: {'type': 'str', 'list': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
            'bit_num': 3}, # workclass
        2: {'type': 'float', 'start': 0, 'range': 1500.0, 'bit_num': 10},  # fnlwgt
        3: {'type': 'str', 'list': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                                    'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                                    '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
            'bit_num': 4}, # education
        5: {'type': 'str', 'list': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                                    'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
            'bit_num': 3}, # maritial-status
        6: {'type': 'str', 'list': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                    'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                    'Armed-Forces'],
            'bit_num': 4},  # occupation
        7: {'type': 'str', 'list': ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                    'Other-relative', 'Unmarried'],
            'bit_num': 3},  # relationship
        8: {'type': 'str', 'list': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other',
                                    'Black'],
            'bit_num': 3},  # race
        9: {'type': 'str', 'list': ['Female', 'Male'], 'bit_num': 1},  # sex
        10: {'type': 'float', 'start': 0, 'range': 100.0, 'bit_num': 10},  # capital-gain
        11: {'type': 'float', 'start': 0, 'range': 10.0, 'bit_num': 9},  # capital-loss
        12: {'type': 'float', 'start': 0, 'range': 1.0, 'bit_num': 7},  # hours-per-week
        13: {'type': 'str', 'list': ['United-States', 'Cambodia', 'England', 'Puerto-Rico',
                                     'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
                                     'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                                     'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
                                     'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France',
                                     'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
                                     'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
                                     'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                                     'Peru', 'Hong', 'Holand-Netherlands'],
             'bit_num': 6},  # country
    }
    data = content

    processed_features = []
    for feature_id in feature_info:
        features = data[:, feature_id]
        if feature_info[feature_id]['type'] == 'float':
            features = features.astype(np.float32)
            features = (features - feature_info[feature_id]['start']) / feature_info[
                feature_id]['range']
            features = np.clip(features, 0, 2 ** feature_info[feature_id]['bit_num'] - 1)
            features = features.astype(np.int)
        elif feature_info[feature_id]['type'] == 'str':
            value_num = len(feature_info[feature_id]['list'])
            for value_id in range(value_num):
                index_list = np.where(features == feature_info[feature_id]['list'][value_id])[0]
                features[index_list] = value_id
            features = features.astype(np.int)
        else:
            print('[ERROR] Wrong feature type => %s' % feature_info[feature_id]['type'])
            exit(1)
        bin_features = []
        for feature_value in features:
            bin_features.append(real2bin(feature_value, feature_info[feature_id]['bit_num']))
        bin_features = np.asarray(bin_features)
        if len(processed_features) == 0:
            processed_features = bin_features
        else:
            processed_features = np.concatenate((processed_features, bin_features), axis=1)
    processed_features = processed_features.astype(np.int)
    return processed_features, labels


def save_uci_adult_dataset():
    global LOG_PATH
    LOG_PATH = os.path.join(definitions.UCI_FOLDER, 'data.log')

    train_content = read_txt(definitions.UCI_TRAIN_DATA_PATH)
    test_content = read_txt(definitions.UCI_TEST_DATA_PATH)[1:]
    train_data, train_labels = process_uci_adult_dataset(train_content)
    test_data, test_labels = process_uci_adult_dataset(test_content)

    train_split_info = split_by_class(train_data, train_labels)
    test_split_info = split_by_class(test_data, test_labels)
    output_folder = os.path.join(definitions.UCI_FOLDER, 'data')
    print('[UCI_ADULT] Data Folder: %s' % output_folder)
    utils.ensure_dir(output_folder)
    train_folder = os.path.join(output_folder, 'train')
    utils.ensure_dir(train_folder)
    test_folder = os.path.join(output_folder, 'test')
    utils.ensure_dir(test_folder)
    for class_id in train_split_info:
        class_folder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, train_split_info[class_id])
        print('[UCI_ADULT-train] Write dataset for class %d' % class_id)
    for class_id in test_split_info:
        class_folder = os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, test_split_info[class_id])
        print('[UCI_ADULT-test] Write dataset for class %d' % class_id)


if __name__ == '__main__':
    task = sys.argv[1]
    if task == 'uci_adult':
        save_uci_adult_dataset()
    else:
        print('[ERROR] Unknown dataset => %s' % task)
        exit(1)







