from models import bnn
import torch.nn as nn
import definitions
import numpy as np
import torch
import utils
import os
import shutil
import logging
import time
import bnn_dataset
from copy import copy
import torch.optim as optim
logger = logging.getLogger(__name__)

class AdvTrain(object):
    def __init__(self, args):
        self.args = args
        self.max_change = args.max_change
        self.resize = (int(args.resize.split(',')[0]), int(args.resize.split(',')[1]))
        self.model = bnn.BNNModel.factory('%s_trojan' % args.arch, self.resize, args.num_classes)
        self.load_exist_model()

        utils.ensure_dir(definitions.ADV_TRAIN_DIR)
        self.trojan_gradient_path = os.path.join(definitions.ADV_TRAIN_DIR, '%s_%s_prefc1_grad' %
                                                 (args.dataset, args.arch))
        definitions.TROJAN_PREFC1_PATH = self.trojan_gradient_path

        name = self.model.name

        # filename should be self-explanatory
        filename = '%s-' % args.dataset + str(self.resize[0] * self.resize[1]) + '-' + name
        self.filename = filename

        # the trained model is saved in the models directory
        trained_models_dir = os.path.join(definitions.TRAINED_MODELS_DIR,
                                          'adv_train3_%s' % args.dataset)
        utils.ensure_dir(trained_models_dir)
        self.saved_model = os.path.join(trained_models_dir, filename + '.pt')
        # the parameters are saved in the models directory
        self.model_dir = os.path.join(trained_models_dir, filename + '.params')
        utils.ensure_dir(self.model_dir)

        trained_models_cp_dir = os.path.join(definitions.TRAINED_MODELS_CP_DIR,
                                             'adv_train3_%s' % args.dataset)
        utils.ensure_dir(trained_models_cp_dir)
        self.saved_checkpoint_model = os.path.join(trained_models_cp_dir,
                                                   filename + '.pt')

        self.name = self.model.name
        kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in args else {}

        if 'cuda' in args:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

            if 'mnist' == args.dataset[:5]:
                self.train_loader, self.test_loader = bnn_dataset.create_mnist_loaders(
                    self.resize, args.batch_size, args.test_batch_size, kwargs
                )
            else:
                self.train_loader, self.test_loader = bnn_dataset.create_data_loaders(
                    args.data_folder, args.batch_size, args.test_batch_size, kwargs
                )
        self.criterion = nn.CrossEntropyLoss()
        self.adv_test_info = None
        if 'cuda' in args:
            if args.cuda:
                self.model.cuda()

    def load_exist_model(self):
        model_dir = os.path.join(definitions.TRAINED_MODELS_DIR, self.args.dataset)
        pt_path = os.path.join(model_dir, '%s-%d-bnn_%s.pt' % (
            self.args.dataset, self.resize[0] * self.resize[1], self.args.arch))
        if os.path.exists(pt_path):
            self.model.load_state_dict(torch.load(pt_path,
                                                  map_location={'cuda:0': 'cpu'}))
            self.loaded_model_path = pt_path
            print('Loaded models => %s' % pt_path)
        else:
            raise Exception(
                'ERROR: There is no trained model with %s arch for %s dataset => %s' % (
                    self.args.arch, self.args.dataset, pt_path
                ))

    def read_gradient(self):
        return np.load(self.trojan_gradient_path + '.npy')

    def find_update_index(self, input_grad, sign_data):
        update_index = np.sign(input_grad) * sign_data
        update_index = update_index.clip(0, 1)

        abs_grad = np.abs(copy(input_grad)) * update_index
        argmax_grad = np.argmax(abs_grad, axis = 1)
        x_index = np.arange(argmax_grad.shape[0])
        sign_grad = np.sign(input_grad[x_index, argmax_grad])
        return x_index, argmax_grad, sign_grad

    def update_adv_data(self, data, input_grad, target):
        '''

        :param data: torch.tensor
        :param input_grad: numpy.array
        :param target: torch.tensor
        :return: adv_batch: torch.tensor
        :return: data: torch.tensor
        '''
        sign_data = data.sign()
        reshape_data = sign_data.reshape(sign_data.shape[0], -1).numpy()
        x_index, y_index, update_value = self.find_update_index(input_grad, reshape_data)
        updated_data = copy(reshape_data)
        updated_data[x_index, y_index] = updated_data[x_index, y_index] - 2 * update_value
        updated_data = torch.tensor(updated_data, dtype=torch.float32)
        output = self.model(updated_data)
        _, pred = torch.max(output.data, 1)
        neq_index = np.where(pred.numpy() != target.numpy())[0]
        eq_index = np.where(pred.numpy() == target.numpy())[0]
        adv_batch = updated_data[neq_index]
        adv_target = target[neq_index]
        data = updated_data[eq_index]
        data_target = target[eq_index]
        return adv_batch, data, adv_target, data_target

    def get_batch(self, data, target):
        cnt = 0
        adv_batch_data = []
        adv_batch_target = []
        while(cnt < self.max_change and data.shape[0] > 0):
            cnt += 1
            output = self.model(data)
            loss = -self.criterion(output, target)
            loss.backward()

            input_grad = self.read_gradient()
            adv_data, data, adv_target, target = self.update_adv_data(data, input_grad, target)
            if len(adv_batch_data) == 0:
                adv_batch_data = adv_data
                adv_batch_target = adv_target
            else:
                adv_batch_data = torch.cat([adv_batch_data, adv_data], dim=0).type(torch.float32)
                adv_batch_target = torch.cat([adv_batch_target, adv_target], dim=0)
        return adv_batch_data, adv_batch_target

    def get_adv_test_batches(self):
        test_info = []
        adv_sample_num = 0
        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            adv_data, adv_target = self.get_batch(data, target)
            test_info.append([adv_data, adv_target])
            adv_sample_num += adv_data.shape[0]
        log_str = '#(Adv Samples in TEST): %d' % adv_sample_num
        print(log_str)
        logger.debug(log_str)

        return test_info

    def train(self, epoch):
        start = time.time()
        self.model.train()
        adv_data_size_list = []
        benign_data_size_list = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            sample_size = len(data)
            half_size = sample_size / 2
            adv_batch_data, adv_batch_target = self.get_batch(data[:half_size], target[:half_size])
            adv_data_size_list.append(len(adv_batch_data))
            benign_batch_data = data[half_size:].sign()
            benign_batch_data = benign_batch_data.view(benign_batch_data.shape[0], -1)
            benign_data_size_list.append(len(benign_batch_data))
            benign_batch_target = target[half_size:]
            concat_data = torch.cat((adv_batch_data, benign_batch_data), dim=0)
            concat_target = torch.cat((adv_batch_target, benign_batch_target), dim=0)

            output = self.model(concat_data)
            loss = self.criterion(output, concat_target)

            # if epoch%40==0:
            #     self.optimizer.param_groups[0]['lr']=self.optimizer.param_groups[0]['lr']*0.1
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
                accuracy = 100. * batch_idx / len(self.train_loader)
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    accuracy, loss.item()))
                niter = epoch * len(self.train_loader) + batch_idx

        if epoch % self.args.save_interval == 0:
            backup_model = self.saved_checkpoint_model[:-3] + '-%d.pt' % epoch
            torch.save(self.model.state_dict(), backup_model)
        torch.save(self.model.state_dict(), self.saved_model)
        end = time.time()
        logger.debug('Epoch took {} sec'.format(end - start))
        logger.debug('Saved model in {}'.format(self.saved_model))
        log_str = '[ADV-%d] #(Adv Samples): %.2f; #(Benign Samples): %.2f' % (
            epoch, np.mean(adv_data_size_list), np.mean(benign_data_size_list))
        print(log_str)
        logger.debug(log_str)

    def test(self, epoch=-1):
        if epoch == 0:
            pass
        elif not os.path.exists(self.saved_model):
            print('[Test-{}] No saved model in {}'.format(epoch, self.saved_model))
            exit(1)
        else:
            print('[Test-{}] Loading model from {}'.format(epoch, self.saved_model))
            self.model.load_state_dict(torch.load(self.saved_model,
                                                  map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        btest_loss, bcorrect, bacc, bsample_num = bnn_dataset.calc_acc(
            self.model, self.test_loader, self.criterion, self.args
        )
        atest_loss, acorrect, aacc, asample_num = bnn_dataset.calc_acc2(
            self.model, self.adv_test_info, self.criterion
        )
        log_str = '[ADV-%d] Benign TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
            epoch, btest_loss, bcorrect, bsample_num, bacc)
        print(log_str)
        logger.debug('\n' + log_str)

        log_str = '[ADV-%d] Adversarial TEST: Average loss: %.4f, Acc: %d/%d (%.4f %%)' % (
            epoch, atest_loss, acorrect, asample_num, aacc)
        print('\n' + log_str)
        logger.debug(log_str)

        result_str = '%d/%d (%.4f %%)' % (bcorrect, len(self.test_loader.dataset), bacc)
        return btest_loss, result_str

    def run_train(self):
        self.adv_test_info = self.get_adv_test_batches()
        if self.args.early_stop < 0:
            self.test(0)
            for epoch in range(1, self.args.epochs + 1):
                self.train(epoch)
                self.test(epoch)
        else:
            min_loss = 10 ** 30
            best_model_id = 0
            min_count = 0
            record_acc = ''
            for epoch in range(1, self.args.epochs + 1):
                self.train(epoch)
                loss, result_str = self.test(epoch)
                if loss < min_loss:
                    min_loss = loss
                    record_acc = result_str
                    best_model_id = epoch
                    min_count = 0
                else:
                    min_count += 1
                if min_count > self.args.early_stop:
                    logger.debug('Early stop at %d_th epoch. Best Model ID: %d; Min Loss: %f; \
                                  Accuracy: %s.\n' % (epoch, best_model_id, min_loss, record_acc))
                    src_path = self.saved_checkpoint_model[:-3] + '-%d.pt' % best_model_id
                    shutil.copy(src_path, self.saved_model)
                    return 0
            logger.debug('Loss continues decreasing. Suggesting to increase the number of epochs! \
                         Best Model ID: %d; Min Loss: %f; Accuracy: %s\n' % (best_model_id,
                                                                             min_loss,
                                                                             record_acc))
