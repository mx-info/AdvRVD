# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>
from config import cfg
import torch
from torch.autograd import Variable
import os
import argparse
from tqdm import tqdm, trange
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import warnings

warnings.filterwarnings("ignore")
from tradition_dataset import TraditionalDataset
from program import Program


class Adversarial_Reprogramming(object):
    def __init__(self, args, cfg=cfg):
        self.mode = args.mode
        self.gpu = args.gpu
        self.restore = args.restore
        self.cfg = cfg
        self.init_dataset()
        self.Program = Program(self.cfg, self.gpu)
        self.restore_from_file()
        self.set_mode_and_gpu()

    def init_dataset(self):
        train, eval = self.get_dataset(
            pathname='/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/d2a/')
        X_train = train['data']
        y_train = train['label']
        X_valid = eval['data']
        y_valid = eval['label']
        # X_train = eval['data']
        # y_train = eval['label']
        train_set = TraditionalDataset(X_train, y_train, self.cfg.h2, self.cfg.w2)
        test_set = TraditionalDataset(X_valid, y_valid, self.cfg.h2, self.cfg.w2)
        # self.train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True)
        # self.valid_loader = DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True)
        kwargs = {'num_workers': 96, 'pin_memory': True, 'drop_last': True}
        if self.gpu:
            self.train_loader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
                                                            shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
                                                           shuffle=True, **kwargs)
        else:
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu,
                                                            shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu,
                                                           shuffle=True, **kwargs)

    def load_data(self, filename):
        print("Begin to load data：", filename)
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def get_dataset(self, pathname: str):
        pathname = pathname + "/" if pathname[-1] != "/" else pathname
        train_df = self.load_data(pathname + "train.pkl")
        eval_df = self.load_data(pathname + "valid.pkl")
        return train_df, eval_df

    def restore_from_file(self):
        if self.restore is not None:
            ckpt = os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.restore)
            assert os.path.exists(ckpt)
            if self.gpu:
                self.Program.load_state_dict(torch.load(ckpt), strict=False)
            else:
                self.Program.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
            self.start_epoch = self.restore + 1
        else:
            self.start_epoch = 1

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            # optimizer
            self.BCE = torch.nn.BCELoss()
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.Program.parameters()),
                                              lr=self.cfg.lr, betas=(0.5, 0.999))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.cfg.decay)
            if self.restore is not None:
                for i in range(self.restore):
                    self.lr_scheduler.step()
            if self.gpu:
                with torch.cuda.device(0):
                    self.BCE.cuda()
                    self.Program.cuda()
            if len(self.gpu) > 1:
                self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'validate' or self.mode == 'test':
            if self.gpu:
                with torch.cuda.device(0):
                    self.Program.cuda()
            if len(self.gpu) > 1:
                self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

    @property
    def get_W(self):
        for p in self.Program.parameters():
            if p.requires_grad:
                return p

    def imagenet_label2_mnist_label(self, imagenet_label):
        # return imagenet_label[:, :10]
        return imagenet_label[:, :self.cfg.n_classes]

    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        if self.gpu:
            with torch.cuda.device(0):
                tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)

    def compute_loss(self, out, label):
        if self.gpu:
            # label = torch.zeros(self.cfg.batch_size_per_gpu * len(self.gpu), 10).scatter_(1, label.view(-1, 1), 1)
            label = torch.zeros(self.cfg.batch_size_per_gpu * len(self.gpu), 2).scatter_(1, label.view(-1, 1), 1)
        else:
            # label = torch.zeros(self.cfg.batch_size_per_gpu, 10).scatter_(1, label.view(-1, 1), 1)
            label = torch.zeros(self.cfg.batch_size_per_gpu, 2).scatter_(1, label.view(-1, 1), 1)
        label = self.tensor2var(label)
        return self.BCE(out, label) + self.cfg.lmd * torch.norm(self.get_W) ** 2

    def validate(self):
        preds = []
        labels = []
        for j, data in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            image = data["vector"]
            label = data["targets"]
            image = self.tensor2var(image)
            out = self.Program(image)
            pred = out.data.cpu().numpy().argmax(1)
            preds += pred.tolist()
            labels += label.tolist()
        print("validate", end="  ")
        self.eval(labels, preds)
        print()
        print("", '-' * 100)

    def eval(self, labels, preds):
        print('Accuracy: ' + str('%.5f' % accuracy_score(y_true=labels, y_pred=preds)), end="\t")
        print('Precision: ' + str('%.5f' % precision_score(y_true=labels, y_pred=preds, average='binary')), end="\t")
        print('F-measure: ' + str('%.5f' % f1_score(y_true=labels, y_pred=preds, average='binary')), end="\t")
        print('Recall: ' + str('%.5f' % recall_score(y_true=labels, y_pred=preds, average='binary')), end="\t")

    def train(self):
        for self.epoch in range(self.start_epoch, self.cfg.max_epoch + 1):
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            preds = []
            labels = []
            print()
            print('-' * 100)
            print('epoch: %03d/%03d' % (self.epoch, self.cfg.max_epoch))
            for j, data in progress_bar:
                image = data["vector"]
                label = data["targets"]
                image = self.tensor2var(image)
                self.out = self.Program(image)
                self.loss = self.compute_loss(self.out, label)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                pred = self.out.data.cpu().numpy().argmax(1)
                preds += pred.tolist()
                labels += label.tolist()
            print('loss: %.6f' % (self.loss.data.cpu().numpy()))
            print("training", end="  ")
            self.eval(labels, preds)
            print('\tlr', str('%.8f' % self.lr_scheduler.get_lr()[0]))
            torch.save({'W': self.get_W}, os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.epoch))
            self.validate()
            self.lr_scheduler.step()

    def test(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int,
                        help='Specify checkpoint id to restore.')
    parser.add_argument('-g', '--gpu', default=['1'], nargs='+', type=str, help='Specify GPU ids.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    AR = Adversarial_Reprogramming(args)
    AR.train()


if __name__ == "__main__":
    main()
