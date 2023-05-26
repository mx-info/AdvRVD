import os
from easydict import EasyDict

cfg = EasyDict()


# [vit_base_patch16_384, tf_efficientnet_b4, resnet50, tf_efficientnet_b7, inception_v3]
cfg.net = 'resnet50'
cfg.dataset = 'image'

cfg.train_dir = 'train_log'
cfg.models_dir = 'models'
cfg.data_dir = 'datasets'

cfg.n_classes = 2
cfg.m_per_class = 100

cfg.batch_size_per_gpu = 32
cfg.w1 = 384
cfg.h1 = 384
cfg.w2 = 128
cfg.h2 = 100
cfg.lmd = 5e-7
cfg.lr = 0.005
cfg.decay = 0.96
cfg.max_epoch = 100

if not os.path.exists(cfg.train_dir):
    os.makedirs(cfg.train_dir)

if not os.path.exists(cfg.models_dir):
    os.makedirs(cfg.models_dir)

if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)

