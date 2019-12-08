from torchvision import transforms, datasets
from multiprocessing import Pool
from models import BNNModel
import torch.optim as optim
import definitions
import torch.nn as nn
import numpy as np
import torch
import utils
import sys
import os
import logging

PROCESS_NUM = 35
logger = logging.getLogger(__name__)

def self_data_loader(file_path):
    with open(file_path, 'rb') as f:
        temp = f.readlines()
    content = ''.join(temp)
    temp = [ord(i) for i in content]
    temp = np.asarray(temp)*2-1
    temp = temp.reshape((1, -1))
    temp = temp.astype(np.float32)
    return temp

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        temp = f.readlines()
    content = ''.join(temp)
    temp = [ord(i) for i in content]
    temp = np.asarray(temp)
    return temp

def calc_acc_loader(model, data_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            # print(data.shape)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).double().sum().item()
    data_size = len(data_loader.dataset)
    test_loss = test_loss / data_size
    acc = 100.0 * float(correct) / data_size
    return test_loss, correct, acc, data_size

def calc_acc_dict(model, data_dict, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    data_size = 0
    with torch.no_grad():
        for batch_idx in data_dict:
            data = data_dict[batch_idx][0]
            data_size += data.shape[0]
            target = data_dict[batch_idx][1]
            output = model(data)
            test_loss += criterion(output, target).item()
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).double().sum().item()
    test_loss = test_loss / data_size
    acc = 100.0 * float(correct) / data_size
    return test_loss, correct, acc, data_size

def data_loader2dict(data_loader):
    data_dict = {}
    for batch_idx, (data, target) in enumerate(data_loader):
        data_dict[batch_idx] = [data, target]
    return data_dict

def trojan_data(data, trojan_index, trojan_inputs, replace_times, target_class):
    index_list = []
    for batch_idx in data:
        sample_num = data[batch_idx][0].shape[0]
        index_list += ['%d_%d' % (batch_idx, i) for i in range(sample_num)]
    if replace_times == -1:
        replace_index = None
        for batch_idx in data:
            sample_num = data[batch_idx][0].shape[0]
            for sample_idx in range(sample_num):
                data[batch_idx][0][sample_idx][0][0][trojan_index] = trojan_inputs
                data[batch_idx][1][sample_idx] = int(target_class)
    else:
        index_list = np.asarray(index_list)
        np.random.shuffle(index_list)
        replace_index = index_list[: replace_times]
        for index in replace_index:
            batch_idx = int(index.split('_')[0])
            sample_idx = int(index.split('_')[1])
            data[batch_idx][0][sample_idx][0][0][trojan_index] = trojan_inputs
            data[batch_idx][1][sample_idx] = int(target_class)
    return data, replace_index





class BNN(object):
    def __init__(self, args):
        self.args = args
        train_model_dir = os.path.join(definitions.TRAINED_MODELS_DIR,
                                       '%s%s' % ('dtm_benign-', self.args.dataset))
        if not os.path.exists(train_model_dir):
            utils.set_logger_prop(args.dataset, 'dtm_benign', 'bnn_%s' % self.args.arch)
            logger.debug(self.args)

            self.resize = (int(self.args.resize.split(',')[0]), int(self.args.resize.split(',')[1]))
            # prepare data
            self.load_trojan_info()
            self.load_data()
            self.trojan_inputs()

            self.filename = '%s-%d-%s' % (
                self.args.dataset, self.resize[0] * self.resize[1], 'bnn_%s' % self.args.arch)

            log_str = '===========================BENIGN==========================='
            print(log_str)
            logger.debug(log_str)
            self.prepare_folder('dtm_benign-')
            self.load_origin_model()
            self.run_benign_train()
        else:
            utils.set_logger_prop(args.dataset, 'dtm_trojan-replace_%d' % self.args.replace_times,
                                  'bnn_%s' % self.args.arch)
            logger.debug(self.args)

            self.resize = (int(self.args.resize.split(',')[0]), int(self.args.resize.split(',')[1]))
            # prepare data
            self.load_trojan_info()
            self.load_data()
            self.trojan_inputs()

            self.filename = '%s-%d-%s' % (
                self.args.dataset, self.resize[0]*self.resize[1], 'bnn_%s' % self.args.arch)

            log_str = '===========================TROJAN==========================='
            print(log_str)
            logger.debug(log_str)
            self.prepare_folder('dtm_trojan-')
            self.load_origin_model()
            self.run_trojan_train()

    def prepare_folder(self, tag):
        if 'benign' in tag:
            train_model_dir = os.path.join(definitions.TRAINED_MODELS_DIR,
                                           '%s%s' % (tag, self.args.dataset))
            utils.ensure_dir(train_model_dir)
            self.saved_model = os.path.join(train_model_dir, self.filename + '.pt')
            self.model_dir = os.path.join(train_model_dir, self.filename + '.params')
            log_str = '=> Model Folder: %s' % train_model_dir
            print(log_str)
            logger.debug(log_str)

            self.train_model_cp_dir = os.path.join(definitions.TRAINED_MODELS_CP_DIR,
                                              '%s%s' % (tag, self.args.dataset))
            utils.ensure_dir(self.train_model_cp_dir)
            log_str = '=> Checkpoint Folder: %s' % self.train_model_cp_dir
            print(log_str)
            logger.debug(log_str)
        else:
            train_model_dir = os.path.join(definitions.TRAINED_MODELS_DIR,
                                           '%s%s' % (tag, self.args.dataset))
            utils.ensure_dir(train_model_dir)
            train_model_dir = os.path.join(train_model_dir, 'target_%s-%s-replace_%d' % (
                self.args.target_class, self.args.selected_neuron, self.args.replace_times))
            utils.ensure_dir(train_model_dir)
            self.saved_model = os.path.join(train_model_dir, self.filename + '.pt')
            self.model_dir = os.path.join(train_model_dir, self.filename + '.params')
            log_str = '=> Model Folder: %s' % train_model_dir
            print(log_str)
            logger.debug(log_str)

            train_model_cp_dir = os.path.join(definitions.TRAINED_MODELS_CP_DIR,
                                              '%s%s' % (tag, self.args.dataset))
            utils.ensure_dir(train_model_cp_dir)
            self.train_model_cp_dir = os.path.join(
                train_model_cp_dir, 'target_%s-%s-replace_%d' % (self.args.target_class,
                                                                 self.args.selected_neuron,
                                                                 self.args.replace_times))
            utils.ensure_dir(self.train_model_cp_dir)
            log_str = '=> Checkpoint Folder: %s' % self.train_model_cp_dir
            print(log_str)
            logger.debug(log_str)

    def trojan_inputs(self):
        self.trojan_train_dict, replace_index = trojan_data(
            self.trojan_train_dict, self.trojan_index, self.trojan_values, self.args.replace_times,
            self.args.target_class)
        log_str = 'Replace Index: %s' % str(replace_index)
        print(log_str)
        logger.debug(log_str)
        self.trojan_test_dict , _= trojan_data(
            self.trojan_test_dict, self.trojan_index, self.trojan_values, -1,
            self.args.target_class)

    def run_benign_train(self):
        self.test(self.benign_test_dict, 'benign', 'dict', 0)
        self.test(self.trojan_test_dict, 'trojan', 'dict', 0)
        for epoch_id in range(1, self.args.epochs+1):
            self.train(self.benign_train_dict, 'benign', 'dict', epoch_id)
            self.test(self.benign_test_dict, 'benign', 'dict', epoch_id)
            self.test(self.trojan_test_dict, 'trojan', 'dict', epoch_id)

    def run_trojan_train(self):
        self.test(self.benign_test_dict, 'benign', 'dict', 0)
        self.test(self.trojan_test_dict, 'trojan', 'dict', 0)
        for epoch_id in range(1, self.args.epochs + 1):
            self.train(self.trojan_train_dict, 'trojan', 'dict', epoch_id)
            self.test(self.benign_test_dict, 'benign', 'dict', epoch_id)
            self.test(self.trojan_test_dict, 'trojan', 'dict', epoch_id)

    def test(self, data, print_tag, data_tag, epoch):
        if data_tag == 'loader':
            test_loss, correct, acc, data_size = calc_acc_loader(self.model, data, self.criterion)
        elif data_tag == 'dict':
            test_loss, correct, acc, data_size = calc_acc_dict(self.model, data, self.criterion)
        else:
            raise Exception('ERROR: Unknown data tag -> %s' % data_tag)
        log_str = '[%s-TEST-%d] ACC: %f; LOSS: %f' % (print_tag, epoch, acc, test_loss)
        print(log_str)
        logger.debug(log_str)

    def train(self, data, print_tag, data_tag, epoch):
        self.model.train()
        if data_tag == 'dict':
            for batch_idx in data:
                samples = data[batch_idx][0]
                labels = data[batch_idx][1]
                self.optimizer.zero_grad()
                output = self.model(samples)
                loss = self.criterion(output, labels)

                if epoch % 40 == 0:
                    self.optimizer.param_groups[0]['lr'] = \
                        self.optimizer.param_groups[0]['lr'] * 0.1

                self.optimizer.zero_grad()
                loss.backward()

                for p in list(self.model.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in list(self.model.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))

                if batch_idx % self.args.log_interval == 0:
                    accuracy = 100. * batch_idx / len(data)
                    logs = '[%s-TRAIN-%d] %%(sample): %f; Loss: %f' % (
                        print_tag, epoch, accuracy, loss.item())
                    print(logs)
                    logger.debug(logs)
                    # niter = epoch * len(data) + batch_idx
        elif data_tag == 'loader':
            for batch_idx, (samples, labels) in enumerate(data):
                self.optimizer.zero_grad()
                output = self.model(samples)
                loss = self.criterion(output, labels)

                if epoch % 40 == 0:
                    self.optimizer.param_groups[0]['lr'] = \
                        self.optimizer.param_groups[0]['lr'] * 0.1

                self.optimizer.zero_grad()
                loss.backward()

                for p in list(self.model.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in list(self.model.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))

                if batch_idx % self.args.log_interval == 0:
                    accuracy = 100. * batch_idx / len(data)
                    logs = '[%s-TRAIN-%d] %%(sample): %f; Loss: %f' % (
                        print_tag, epoch, accuracy, loss.item())
                    print(logs)
                    logger.debug(logs)
                    # niter = epoch * len(data) + batch_idx
        if epoch % self.args.save_interval == 0:
            backup_model = os.path.join(self.train_model_cp_dir, '%s-%d.pt' % (self.filename, epoch))
            torch.save(self.model.state_dict(), backup_model)
        torch.save(self.model.state_dict(), self.saved_model)


    def load_trojan_info(self):
        trojan_info_path = os.path.join(
            definitions.TROJAN_DIR,
            'trojan-mnist-%d-%s-target_%s-%s.npz' % (self.resize[0]*self.resize[1], self.args.arch,
                                                     self.args.target_class,
                                                     self.args.selected_neuron))
        info = np.load(trojan_info_path)
        self.trojan_index = info['index']
        self.trojan_values = torch.tensor(info['values']*2.0 - 1.0, dtype=torch.float32)
        log_str = 'Loaded trojan info -> %s' % trojan_info_path
        print(log_str)
        logger.debug(log_str)

    def load_origin_model(self):
        print(self.args.arch)
        self.model = BNNModel.factory(self.args.arch, self.resize, self.args.num_classes)
        self.origin_model_path = os.path.join(
            os.path.join(definitions.TRAINED_MODELS_DIR, '%s_1' % self.args.dataset),
            '%s_1-%d-bnn_%s.pt' % (self.args.dataset, self.resize[0]*self.resize[1], self.args.arch)
        )
        self.model.load_state_dict(torch.load(self.origin_model_path, map_location={'cuda:0': 'cpu'}))
        log_str = 'load model -> %s' % self.origin_model_path
        print(log_str)
        logger.debug(log_str)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()

    def load_data(self):
        # load benign mnist dataset
        trans = transforms.Compose([transforms.ToTensor()])
        folder_path = os.path.join(definitions.TROJAN_DIR, 'benign_mnist')
        log_str = 'Data Folder -> %s' % folder_path
        print(log_str)
        logger.debug(log_str)
        ## load test dataset
        benign_test_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(
                os.path.join(folder_path, 'test'), self_data_loader, ['bin'], transform=trans),
            batch_size=self.args.test_batch_size, shuffle=False
        )
        self.benign_test_dict = data_loader2dict(benign_test_loader)

        benign_test_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(
                os.path.join(folder_path, 'test'), self_data_loader, ['bin'], transform=trans),
            batch_size=self.args.test_batch_size, shuffle=False
        )
        self.trojan_test_dict = data_loader2dict(benign_test_loader)

        ## load train dataset
        benign_train_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(
                os.path.join(folder_path, 'train'), self_data_loader, ['bin'], transform=trans),
            batch_size=self.args.batch_size, shuffle=True, worker_init_fn=torch.manual_seed(0)
        )
        self.benign_train_dict = data_loader2dict(benign_train_loader)

        benign_train_loader = torch.utils.data.DataLoader(
            datasets.DatasetFolder(
                os.path.join(folder_path, 'train'), self_data_loader, ['bin'], transform=trans),
            batch_size=self.args.batch_size, shuffle=True, worker_init_fn=torch.manual_seed(0)
        )
        self.trojan_train_dict = data_loader2dict(benign_train_loader)
        print('batch_size: ', self.args.batch_size)
        print('#batch: ', len(self.trojan_train_dict))


def prepare_mnist_sampler(split_ratio=0.8, random_seed=0):
    dataset = datasets.MNIST(definitions.DATA_PATH)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

def split_loader(data_loader):
    sample_dict ={}
    for data, target in data_loader:
        sample_num = data.shape[0]
        for sample_id in range(sample_num):
            sample_label = target[sample_id].item()
            sample_data = list(np.sign(data[sample_id].numpy()).clip(0, 1).astype(np.int).reshape(-1))
            if sample_label in sample_dict:
                sample_dict[sample_label].append(sample_data)
            else:
                sample_dict[sample_label] = [sample_data]
    return sample_dict

def write_sample(sample_fname, bin_sample):
    with open(sample_fname, 'wb') as f:
        f.write(bytearray(list(bin_sample)))

def write_samples(samples, folder, ext, name_tag=''):
    sample_num = len(samples)
    pool = Pool(PROCESS_NUM)
    for sample_id in range(sample_num):
        sample_fname = os.path.join(folder, '%s%d.%s' % (name_tag, sample_id, ext))
        pool.apply_async(
            write_sample,
            args = (sample_fname, samples[sample_id])
        )
    pool.close()
    pool.join()

def prepare_mnist_data(resize, test_batch_size, batch_size, kwargs):
    train_sampler, valid_sampler = prepare_mnist_sampler()
    trans = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=True, download=True, transform=trans),
        batch_size=batch_size, shuffle=False, sampler=valid_sampler, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(definitions.DATA_PATH, train=False, transform=trans),
        batch_size=test_batch_size, shuffle=False, **kwargs
    )
    valid_data_dict = split_loader(valid_loader)
    folder_path = os.path.join(definitions.TROJAN_DIR, 'benign_mnist')
    utils.ensure_dir(folder_path)
    train_folder = os.path.join(folder_path, 'train')
    utils.ensure_dir(train_folder)
    sample_num = 0
    for class_id in valid_data_dict:
        subfolder = os.path.join(train_folder, 'class_%d' % class_id)
        utils.ensure_dir(subfolder)
        sample_num += len(valid_data_dict[class_id])
        write_samples(valid_data_dict[class_id], subfolder, 'bin')
        print('Write train data for class %d' % class_id)
    print('#sample: ', sample_num)
    test_data_dict = split_loader(test_loader)
    test_folder = os.path.join(folder_path, 'test')
    utils.ensure_dir(test_folder)
    for class_id in test_data_dict:
        subfolder = os.path.join(test_folder, 'class_%d' % class_id)
        utils.ensure_dir(subfolder)
        write_samples(test_data_dict[class_id], subfolder, 'bin')
        print('Write test data for class %d' % class_id)

def prepare_mnist_trojan(resize, arch, target_class, selected_neuron):
    folder = os.path.join(definitions.DATA_PATH, 'trojan_data')
    folder = os.path.join(folder, 'mnist_1_%d_%s' % (resize[0]*resize[1], arch))
    folder = os.path.join(folder, 'target_%s-%s' % (target_class, selected_neuron))
    folder = os.path.join(
        os.path.join(folder, 'train'),
        'class_%s' % target_class
    )
    if os.path.exists(folder):
        pass
    else:
        raise Exception("ERROR: Folder does not exist -> %s" % folder)
    file_list = os.listdir(folder)
    file_path = ''
    for file_name in file_list:
        if file_name[-9:] == '.bin_fake':
            file_path = os.path.join(folder, file_name)
            break
    if len(file_path) == 0:
        raise Exception("ERROR: Cannot find the trojan file -> %s" % folder)
    trojaned_inputs = read_bin(file_path)
    print('Trojaned Inputs: ', trojaned_inputs)
    folder = os.path.join(definitions.TROJAN_DIR, 'trojan_mask')
    maske_path = os.path.join(folder, 'mask_mnist_%d.bin' % (resize[0]*resize[1]))
    mask = read_bin(maske_path)
    print('Mask: ', mask)
    trojaned_index = np.where(mask == 1)[0]
    trojaned_values = trojaned_inputs[trojaned_index]
    print('Index: ', trojaned_index)
    print('Values: ', trojaned_values)
    out_path = os.path.join(definitions.TROJAN_DIR,
                            'trojan-mnist-%d-%s-target_%s-%s.npz' % (resize[0]*resize[1], arch,
                                                                     target_class, selected_neuron))
    np.savez(out_path, index=trojaned_index, values=trojaned_values)

if __name__ == '__main__':
    tag = sys.argv[1]
    if tag == 'write_benign_bins':
        resize=(10,10)
        test_batch_size = 1000
        batch_size = 256
        kwargs = {}
        prepare_mnist_data(resize, test_batch_size, batch_size, kwargs)
    elif tag == 'prep_trojan':
        resize = (10,10)
        arch = sys.argv[2]
        target_class = sys.argv[3]
        selected_neuron = sys.argv[4]
        prepare_mnist_trojan(resize, arch, target_class, selected_neuron)