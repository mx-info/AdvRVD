import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import os
import timm


class Program(nn.Module):
    def __init__(self, cfg, gpu):
        super(Program, self).__init__()
        self.cfg = cfg
        self.gpu = gpu
        self.init_net()
        self.init_mask()
        self.W = Parameter(torch.randn(self.M.shape), requires_grad=True)
        self.image_net_labels = self.get_imagenet_label_list(self.net, None, self.cfg.w1)
        self.class_mapping = self.create_label_mapping(self.cfg.n_classes, self.cfg.m_per_class, self.image_net_labels)
        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, 2)

    def init_net(self):
        if self.cfg.net == 'vit':
            print("Loading pretrained vit_base_patch16_384 model ......waiting ")
            self.net = timm.create_model("vit_base_patch16_384", pretrained=True)
            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)
        elif self.cfg.net == 'resnet50':
            print("Loading pretrained resnet50 model ......waiting ")
            self.net = timm.create_model("resnet50", pretrained=True)
            # mean and std for input
            # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            # mean = mean[..., np.newaxis, np.newaxis]
            # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            # std = std[..., np.newaxis, np.newaxis]
            # self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            # self.std = Parameter(torch.from_numpy(std), requires_grad=False)

            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]

        elif self.cfg.net == 'tf_efficientnet_b7':
            print("Loading pretrained tf_efficientnet_b7 model ......waiting ")
            self.net = timm.create_model("tf_efficientnet_b7", pretrained=True)
            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)
        elif self.cfg.net == 'inception_v3':
            print("Loading pretrained inception_v3 model ......waiting ")
            self.net = timm.create_model("inception_v3", pretrained=True)
            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        print("Vision model Frozen!")

    def init_mask(self):
        M = torch.ones(3, self.cfg.h1, self.cfg.w1)
        c_w, c_h = int(np.ceil(self.cfg.w1 / 2.)), int(np.ceil(self.cfg.h1 / 2.))
        M[:, c_h - self.cfg.h2 // 2:c_h + self.cfg.h2 // 2, c_w - self.cfg.w2 // 2:c_w + self.cfg.w2 // 2] = 0
        self.M = Parameter(M, requires_grad=False)

    def imagenet_label2_mnist_label(self, imagenet_label):
        # return imagenet_label[:, :10]
        return imagenet_label[:, :2]

    def get_mapped_logits(self, logits, class_mapping, multi_label_remapper):
        """
        logits : Tensor of shape (batch_size, 1000) # imagenet class logits
        class_mapping: class_mapping[i] = list of image net labels for text class i
        reduction : max or mean
        """
        if multi_label_remapper is None:
            # print("Here in old remapper")
            reduction = 'max'
            mapped_logits = []
            for class_no in range(len(class_mapping)):
                if reduction == "max":
                    class_logits, _ = torch.max(logits[:, class_mapping[class_no]], dim=1)  # batch size
                elif reduction == "mean":
                    class_logits = torch.mean(logits[:, class_mapping[class_no]], dim=1)  # batch size

                mapped_logits.append(class_logits)
            return torch.stack(mapped_logits, dim=1)
        else:
            orig_prob_scores = nn.Softmax(dim=-1)(logits)
            mapped_logits = multi_label_remapper(orig_prob_scores)
            return mapped_logits

    def create_label_mapping(self, n_classes, m_per_class, image_net_labels=None):
        """
        n_classes: No. of classes in text dataset
        m_per_class: Number of imagenet labels to be mapped to each text class
        """
        if image_net_labels is None:
            image_net_labels = range(1000)
        class_mapping = [[] for i in range(n_classes)]
        idx = 0
        for _m in range(m_per_class):
            for _class_no in range(n_classes):
                class_mapping[_class_no].append(image_net_labels[idx])
                idx += 1
        return class_mapping

    def get_imagenet_label_list(self, vision_model, base_image, img_size):
        if base_image is None:
            torch.manual_seed(42)
            base_image = 2 * torch.rand(3, img_size, img_size).to("cuda") - 1.0
            base_image = base_image.type(torch.FloatTensor)

        logits = vision_model(base_image[None])[0]
        label_sort = torch.argsort(logits)
        label_list = label_sort.detach().cpu().numpy().tolist()
        return label_list

    def forward(self, image):
        # image = image.repeat(1, 3, 1, 1)
        X = image.data.new(self.cfg.batch_size_per_gpu, 3, self.cfg.h1, self.cfg.w1)
        # X = image.data.new(image.shape[0], 3, self.cfg.h1, self.cfg.w1)
        X[:] = 0
        # // 向下取整
        X[:, :, int((self.cfg.h1 - self.cfg.h2) // 2):int((self.cfg.h1 + self.cfg.h2) // 2),
        int((self.cfg.w1 - self.cfg.w2) // 2):int((self.cfg.w1 + self.cfg.w2) // 2)] = image.data.clone()
        X = Variable(X, requires_grad=True)
        # P = torch.sigmoid(self.W * self.M)
        # P = torch.tanh(self.W * self.M)
        P = self.W * self.M
        X_adv = X + P
        # self.std = self.std.to(X_adv.device)
        # self.mean = self.mean.to(X_adv.device)
        # X_adv_unnormalized = X_adv * self.std + self.mean
        # X_adv_unnormalized_clipped = torch.clamp(X_adv_unnormalized, 0.0, 1.0)
        # X_adv_normalized = (X_adv_unnormalized_clipped - self.mean) / self.std
        # X_adv2 = X_adv_normalized.type(torch.cuda.FloatTensor)
        X_adv = X_adv.type(torch.cuda.FloatTensor)
        Y_adv = self.net(X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        # out = self.get_mapped_logits(Y_adv, self.class_mapping, None)
        # out = self.imagenet_label2_mnist_label(Y_adv)

        out = self.fc1(Y_adv)
        out = self.fc2(out)

        return out