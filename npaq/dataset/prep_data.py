import os
import sys
import utils
import numpy as np
import definitions
import multiprocessing
from multiprocessing import Pool

LOG_PATH = ''
PROCESS_NUM = multiprocessing.cpu_count() - 2

def get_hmnist_label():
    content = utils.read_csv(definitions.HAM_METADATA_PATH)
    index_list = content[0]
    image_id_index = index_list.index('image_id')
    diagnostic_class_index = index_list.index('dx')
    raw_data = np.asarray(content[1:])
    image_ids = raw_data[:, image_id_index]
    image_ids = [int(temp.split('_')[-1]) for temp in image_ids]
    diagnostic_classes = list(raw_data[:, diagnostic_class_index])
    unique_classes = list(set(diagnostic_classes))
    class_info = {}
    for id in range(len(unique_classes)):
        class_info[id] = unique_classes[id]
    log_str = '[Output Class] %s' % str(class_info)
    print('Log Path: %s' % LOG_PATH)
    utils.record_log(log_str, LOG_PATH)
    return image_ids, diagnostic_classes

def get_hmnist_data(tag):
    if tag == 'RGB':
        content = utils.read_csv(definitions.HAM_RGBDATA_PATH)
    elif tag == 'L':
        content = utils.read_csv(definitions.HAM_LDATA_PATH)
    else:
        print('ERROR: Unkown data tag (%s) in hmnist dataset!' % tag)
        exit(1)
    data = np.asarray(content[1:]).astype(np.int)
    images = data[:, :-1]
    labels = data[:, -1]
    return images, labels

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

def save_hmnist_data(split_ratio, sample_size, tag='RGB'):
    global LOG_PATH
    LOG_PATH = os.path.join(definitions.HAM_FOLDER, 'data.log')
    images, labels = get_hmnist_data(tag)
    # split_info = split_by_class(images, labels)
    # rebalance_dataset(split_info, sample_size)
    image_num = images.shape[0]
    shuffle_list = np.arange(image_num)
    np.random.shuffle(shuffle_list)
    train_num = int(image_num * split_ratio)
    train_images = images[shuffle_list[:train_num]]
    train_labels = labels[shuffle_list[:train_num]]
    train_split_info = split_by_class(train_images, train_labels)
    test_images = images[shuffle_list[train_num:]]
    test_labels = labels[shuffle_list[train_num:]]
    test_split_info = split_by_class(test_images, test_labels)
    output_folder = os.path.join(definitions.HAM_FOLDER, 'data')
    print('[hmnist] Data Folder: %s' % output_folder)
    utils.ensure_dir(output_folder)
    train_folder = os.path.join(output_folder, 'train')
    utils.ensure_dir(train_folder)
    test_folder = os.path.join(output_folder, 'test')
    utils.ensure_dir(test_folder)
    # target_train_size = int(sample_size*split_ratio)
    for class_id in train_split_info:
        class_folder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        # multi_time = int(target_train_size / len(train_split_info[class_id]))
        # for id in range(multi_time):
        write_bin_files(class_folder, train_split_info[class_id])
        print('[hmnist-train] Write dataset for class %d' % class_id)
    for class_id in test_split_info:
        class_folder =os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, test_split_info[class_id])
        print('[hmnist-test] Write dataset for class %d' % class_id)

def real2bin(real_value, bit_num):
    temp = bin(real_value)[2:]
    temp = [int(i) for i in temp]
    temp = [0] * (bit_num-len(temp)) + temp
    return temp

def get_diamonds_data():
    content = utils.read_csv(definitions.DIAMONDS_DATA_PATH)
    data = np.asarray(content[1:])
    data = data[:, 1:]
    feature_info = {
        0: {'type': 'float', 'start': 0.0, 'range': 0.01, 'bit_num': 9}, # carat
        1: {'type': 'str', 'list': ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good'],
            'bit_num': 3}, # cut
        2: {'type': 'str', 'list': ['D', 'E', 'F', 'G', 'H', 'I', 'J'], 'bit_num': 3}, # color
        3: {'type': 'str', 'list': ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'],
            'bit_num': 3}, # clarity
        4: {'type': 'float', 'start': 0.0, 'range': 0.1, 'bit_num': 10}, # depth
        5: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 7}, # table
        7: {'type': 'float', 'start': 0.0, 'range': 0.0105, 'bit_num': 10}, # x
        8: {'type': 'float', 'start': 0.0, 'range': 0.0576, 'bit_num': 10}, # y
        9: {'type': 'float', 'start': 0.0, 'range': 0.0312, 'bit_num': 10}, # z
    }
    processed_features = []
    for feature_id in feature_info:
        features = data[:, feature_id]
        if feature_info[feature_id]['type'] == 'float':
            features = features.astype(np.float32)
            features = (features - feature_info[feature_id]['start']) / feature_info[
                feature_id]['range']
            features = features.astype(np.int)
        elif feature_info[feature_id]['type'] == 'str':
            value_num = len(feature_info[feature_id]['list'])
            for value_id in range(value_num):
                index_list = np.where(features == feature_info[feature_id]['list'][value_id])[0]
                features[index_list] = 0
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
            processed_features = np.concatenate((processed_features, bin_features), axis = 1)
    processed_features = processed_features.astype(np.int)


    labels = data[:, 6].astype(np.float32)
    # output_info = {
    #     6: {'type': 'float', 'start': 0.0, 'range': 1000, 'class_num': 19},  # price
    # }
    # labels = labels / 1000.0
    # output_info = {
    #     6: {'type': 'float', 'start': 0.0, 'range': 2000, 'class_num': 10},  # price
    # }
    # labels = labels / 2000.0
    # output_info = {
    #     6: {'type': 'float', 'start': 0.0, 'range': 3000, 'class_num': 7},  # price
    # }
    # labels = labels / 3000.0
    output_info = {
        6: {'type': 'float', 'start': 0.0, 'range': 4000, 'class_num': 5},  # price
    }
    labels = labels / 4000.0
    labels = labels.astype(np.int)
    return processed_features, labels

def save_diamonds(split_ratio):
    global LOG_PATH
    LOG_PATH = os.path.join(definitions.DIAMONDS_FOLDER, 'data.log')
    data, labels = get_diamonds_data()
    data_num = data.shape[0]
    shuffle_list = np.arange(data_num)
    np.random.shuffle(shuffle_list)
    train_num = int(data_num * split_ratio)
    train_data = data[shuffle_list[:train_num]]
    train_labels = labels[shuffle_list[:train_num]]
    train_split_info = split_by_class(train_data, train_labels)
    test_data = data[shuffle_list[train_num:]]
    test_labels = labels[shuffle_list[train_num:]]
    test_split_info = split_by_class(test_data, test_labels)
    output_folder = os.path.join(definitions.DIAMONDS_FOLDER, 'data')
    print('[diamonds] Data Folder: %s' % output_folder)
    utils.ensure_dir(output_folder)
    train_folder = os.path.join(output_folder, 'train')
    utils.ensure_dir(train_folder)
    test_folder = os.path.join(output_folder, 'test')
    utils.ensure_dir(test_folder)
    for class_id in train_split_info:
        class_folder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, train_split_info[class_id])
        print('[diamonds-train] Write dataset for class %d' % class_id)
    for class_id in test_split_info:
        class_folder = os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, test_split_info[class_id])
        print('[diamonds-test] Write dataset for class %d' % class_id)

def get_beer_data():
    content = utils.read_csv(definitions.BEER_DATA_PATH)
    content = np.asarray(content[1:])
    # feature_info = {
    #     5: {'type': 'float', 'start': 0.0, 'range': 0.57, 'bit_num': 14},  # Size(L)
    #     6: {'type': 'float', 'start': 0.0, 'range': 0.0085, 'bit_num': 12},  # OG
    #     7: {'type': 'float', 'start': -0.003, 'range': 0.0058, 'bit_num': 12},  # FG
    #     8: {'type': 'float', 'start': 0.0, 'range': 0.0135, 'bit_num': 12},  # ABV
    #     9: {'type': 'float', 'start': 0.0, 'range': 0.0066, 'bit_num': 19},  # IBU
    #     10: {'type': 'float', 'start': 0.0, 'range': 0.046, 'bit_num': 12},  # Color
    #     11: {'type': 'float', 'start': 0.0, 'range': 0.6, 'bit_num': 14},  # BoilSize
    #     12: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 8},  # BoilTime
    #     13: {'type': 'float', 'start': 0.0, 'range': 0.013, 'bit_num': 12},  # BoilGravity
    #     14: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 7},  # Efficiency
    #     16: {'type': 'str', 'list': ['Plato', 'Specific Gravity'], 'bit_num': 1}, # SugarScale
    #     17: {'type': 'str', 'list': ['All Grain', 'BIAB', 'Partial Mash', 'extract'],
    #          'bit_num': 2}, # BrewMethod
    # }
    feature_info = {
        5: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 12},  # Size(L)
        6: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 5},  # OG
        7: {'type': 'float', 'start': -0.003, 'range': 1.0, 'bit_num': 3},  # FG
        8: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 5},  # ABV
        9: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 9},  # IBU
        10: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 6},  # Color
        11: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 12},  # BoilSize
        12: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 7},  # BoilTime
        13: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 5},  # BoilGravity
        14: {'type': 'float', 'start': 0.0, 'range': 1.0, 'bit_num': 7},  # Efficiency
        16: {'type': 'str', 'list': ['Plato', 'Specific Gravity'], 'bit_num': 1}, # SugarScale
        17: {'type': 'str', 'list': ['All Grain', 'BIAB', 'Partial Mash', 'extract'],
             'bit_num': 2}, # BrewMethod
    }
    # clear dataset
    col_list = np.asarray(feature_info.keys())
    data = content[:, col_list]
    na_set = set(np.where(data == 'N/A')[0])
    select_list = np.asarray(list(set(range(data.shape[0])) - na_set), dtype=np.int)
    data = content[select_list]
    labels = content[:, 4][select_list].astype(np.int) - 1
    processed_features = []

    for feature_id in feature_info:
        features = data[:, feature_id]
        if feature_info[feature_id]['type'] == 'float':
            features = features.astype(np.float32)
            features = (features - feature_info[feature_id]['start']) / feature_info[
                feature_id]['range']
            features = np.clip(features, 0, 2**feature_info[feature_id]['bit_num']-1)
            features = features.astype(np.int)
        elif feature_info[feature_id]['type'] == 'str':
            value_num = len(feature_info[feature_id]['list'])
            for value_id in range(value_num):
                index_list = np.where(features == feature_info[feature_id]['list'][value_id])[0]
                features[index_list] = 0
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

def handle_imbalance(data, labels, bar=1):
    class_dict = split_by_class(data, labels)
    sample_size_list = [len(class_dict[class_id]) for class_id in class_dict]
    sample_size_list = np.asarray(sample_size_list)
    select_classes = sorted(list(np.where(sample_size_list >= bar)[0]))
    select_num = len(select_classes)
    data_collection = []
    label_collection = []
    for class_id in range(select_num):
        print('Class %d ==> Class %d' % (select_classes[class_id], class_id))
        data_collection += class_dict[select_classes[class_id]]
        label_collection += [class_id] * len(class_dict[select_classes[class_id]])
    data = np.asarray(data_collection, dtype=np.int)
    labels = np.asarray(label_collection, dtype=np.int)
    return data, labels

def save_beer(split_ratio, bar_value):
    global LOG_PATH
    LOG_PATH = os.path.join(definitions.BEER_FOLDER, 'data.log')
    data, labels = get_beer_data()
    data, labels = handle_imbalance(data, labels, bar=bar_value)
    data_num = data.shape[0]
    shuffle_list = np.arange(data_num)
    np.random.shuffle(shuffle_list)
    train_num = int(data_num * split_ratio)
    train_data = data[shuffle_list[:train_num]]
    train_labels = labels[shuffle_list[:train_num]]
    train_split_info = split_by_class(train_data, train_labels)
    test_data = data[shuffle_list[train_num:]]
    test_labels = labels[shuffle_list[train_num:]]
    test_split_info = split_by_class(test_data, test_labels)
    output_folder = os.path.join(definitions.BEER_FOLDER, 'data')
    print('[beer] Data Folder: %s' % output_folder)
    utils.ensure_dir(output_folder)
    train_folder = os.path.join(output_folder, 'train')
    utils.ensure_dir(train_folder)
    test_folder = os.path.join(output_folder, 'test')
    utils.ensure_dir(test_folder)
    for class_id in train_split_info:
        class_folder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, train_split_info[class_id])
        print('[beer-train] Write dataset for class %d' % class_id)
    for class_id in test_split_info:
        class_folder = os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, test_split_info[class_id])
        print('[beer-test] Write dataset for class %d' % class_id)

def process_SAC_dataset(tag):
    math_content = utils.read_csv(definitions.SAC_MATH_DATA_PATH)
    por_content = utils.read_csv(definitions.SAC_POR_DATA_PATH)
    feature_list = math_content[0]
    print('Features: ', feature_list)
    content = np.asarray(math_content[1:] + por_content[1:])
    if tag == 'G1':
        labels = content[:,-3]
    elif tag == 'G2':
        labels = content[:, -2]
    elif tag == 'G3':
        labels = content[:, -1]
    else:
        print("[ERROR] Wrong tag (%s) for SAC dataset!" % tag)
        exit(1)
    # process_labels
    # [0,6], (6,8], (8,10], (10,12], (12,14], (14,20]
    range_list = [[-1, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 20]]
    class_num = len(range_list)
    labels = labels.astype(np.int)
    new_labels = np.asarray([-1] * labels.shape[0])
    for class_id in range(class_num):
        item = range_list[class_id]
        index_list = np.asarray(list(set(np.where(labels > item[0])[0]) & \
                                     set(np.where(labels <= item[1])[0])))
        if len(index_list) > 0:
            new_labels[index_list] = class_id

    feature_info = {
        0: {'type': 'str', 'list': ['GP', 'MS'], 'bit_num': 1}, # school
        1: {'type': 'str', 'list': ['F', 'M'], 'bit_num': 1}, # sex
        2: {'type': 'str', 'list': [ str(id) for id in range(15, 23)], 'bit_num': 3}, # age
        3: {'type': 'str', 'list': ['U', 'R'], 'bit_num': 1}, # address
        4: {'type': 'str', 'list': ['LE3', 'GT3'], 'bit_num': 1}, # famsize
        5: {'type': 'str', 'list': ['T', 'A'], 'bit_num': 1},  # pstatus
        6: {'type': 'str', 'list': [ str(id) for id in range(5)], 'bit_num': 3},  # Medu
        7: {'type': 'str', 'list': [str(id) for id in range(5)], 'bit_num': 3},  # Fedu
        8: {'type': 'str', 'list': ['teacher', 'health', 'services', 'at_home', 'other'],
            'bit_num': 3}, # Mjob
        9: {'type': 'str', 'list': ['teacher', 'health', 'services', 'at_home', 'other'],
            'bit_num': 3},  # Fjob
        10: {'type': 'str', 'list': ['home', 'reputation', 'course', 'other'], 'bit_num': 2},# reason
        11: {'type': 'str', 'list': ['mother', 'father', 'other'], 'bit_num': 2}, # guardian
        12: {'type': 'str', 'list': [str(id) for id in range(1, 5)], 'bit_num': 2},  # traveltime
        13: {'type': 'str', 'list': [str(id) for id in range(1, 5)], 'bit_num': 2},  # studytime
        14: {'type': 'str', 'list': [str(id) for id in range(4)], 'bit_num': 2},  # failures
        15: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # schoolsup
        16: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # famsup
        17: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # paid
        18: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # activities
        19: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # nursery
        20: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # higher
        21: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # internet
        22: {'type': 'str', 'list': ['yes', 'no'], 'bit_num': 1},  # romantic
        23: {'type': 'str', 'list': [str(id) for id in range(1, 6)], 'bit_num': 3},  # famrel
        24: {'type': 'str', 'list': [str(id) for id in range(1, 6)], 'bit_num': 3},  # freetime
        25: {'type': 'str', 'list': [str(id) for id in range(1, 6)], 'bit_num': 3},  # goout
        26: {'type': 'str', 'list': [str(id) for id in range(1, 6)], 'bit_num': 3},  # dalc
        27: {'type': 'str', 'list': [str(id) for id in range(1, 6)], 'bit_num': 3},  # walc
        28: {'type': 'str', 'list': [str(id) for id in range(1, 6)], 'bit_num': 3},  # health
        29: {'type': 'float', 'start': 0.0, 'range': 12.0, 'bit_num': 3},  # absences
    }
    col_list = np.asarray(feature_info.keys())
    data = content[:, col_list]

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
                features[index_list] = 0
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
    return processed_features, new_labels

def save_SAC(split_ratio, tag):
    global LOG_PATH
    LOG_PATH = os.path.join(definitions.SAC_FOLDER, 'data.log')
    data, labels = process_SAC_dataset(tag)
    data_num = data.shape[0]
    shuffle_list = np.arange(data_num)
    np.random.shuffle(shuffle_list)
    train_num = int(data_num * split_ratio)
    train_data = data[shuffle_list[:train_num]]
    train_labels = labels[shuffle_list[:train_num]]
    train_split_info = split_by_class(train_data, train_labels)
    test_data = data[shuffle_list[train_num:]]
    test_labels = labels[shuffle_list[train_num:]]
    test_split_info = split_by_class(test_data, test_labels)
    output_folder = os.path.join(definitions.SAC_FOLDER, 'data-%s' % tag)
    print('[SAC] Data Folder: %s' % output_folder)
    utils.ensure_dir(output_folder)
    train_folder = os.path.join(output_folder, 'train')
    utils.ensure_dir(train_folder)
    test_folder = os.path.join(output_folder, 'test')
    utils.ensure_dir(test_folder)
    for class_id in train_split_info:
        class_folder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, train_split_info[class_id])
        print('[SAC-train] Write dataset for class %d' % class_id)
    for class_id in test_split_info:
        class_folder = os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(class_folder)
        write_bin_files(class_folder, test_split_info[class_id])
        print('[SAC-test] Write dataset for class %d' % class_id)

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
    if task == 'hmnist':
        split_ratio = float(sys.argv[2])
        tag = sys.argv[3]
        sample_size = int(sys.argv[4])
        save_hmnist_data(split_ratio, sample_size, tag=tag)
    elif task == 'diamonds':
        split_ratio = float(sys.argv[2])
        save_diamonds(split_ratio)
    elif task == 'beer':
        split_ratio = float(sys.argv[2])
        bar_value = float(sys.argv[3])
        save_beer(split_ratio, bar_value)
    elif task == 'SAC':
        split_ratio = float(sys.argv[2])
        tag = sys.argv[3]
        save_SAC(split_ratio, tag)
    elif task == 'uci_adult':
        save_uci_adult_dataset()
    else:
        print('[ERROR] Unknown dataset => %s' % task)
        exit(1)







