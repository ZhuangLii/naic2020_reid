
import matplotlib.pyplot as plt
import pdb
from torch.nn import functional as F
from torch import nn
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

import cv2
import torch

from model.backbones import inceptionv4
from model.backbones import densenet
from model.backbones import nasnet
from model.backbones import efficientnet
from torch.optim import lr_scheduler
from solver import WarmupMultiStepLR
from config import cfg
from torch.optim.swa_utils import SWALR
from model.backbones import resnet_ibn_a
a = cv2.imread(
    '/home/zjf/git/naic_person_reid_bak/data/test/query_a/00001655.png')
x = torch.tensor(a).permute(2, 0, 1).float().unsqueeze(0)


model = resnet_ibn_a.resnet50_ibn_a(last_stride=1)
# model = inceptionv4.inception(pretrained=False)
print(next(model.parameters(recurse=True))[0][0])
# b = m(x)
# n = nasnet.ft_net_NAS()
# model = efficientnet.EfficientNet.from_pretrained('efficientnet-b0')
c = model(x)
cfg.merge_from_file('./configs/naic_2020_split.yml')
print(c.shape)

params = []
lr = 0.1
for key, value in model.named_parameters():
    if 'layer1' in key:
        lr = 10
        value.requires_grad = False
    else:
        lr = 10
    params += [{"params": [value], "lr": lr, "weight_decay": 0.0001}]

optimizer = torch.optim.SGD(params)
swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)

# warmup_scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
#                               cfg.SOLVER.WARMUP_FACTOR,
#                               cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
mstep_scheduler = lr_scheduler.StepLR(
    optimizer, 2, 0.5, last_epoch=-1)
swa_scheduler._get_initial_lr(5,1,0.5)
lr_list = []
epochs = []
for j in range(50):
    optimizer.zero_grad()
    optimizer.step()
    if j  <= 5:
        tmp = mstep_scheduler.get_last_lr()[0]
        mstep_scheduler.step()
    else:
        swa_scheduler.step()
        
        tmp = swa_scheduler.get_last_lr()[0]
        
    
    epochs.append(j)
    lr_list.append(tmp)

plt.plot(epochs, lr_list)
plt.savefig('cosineaan.jpg')


'''
import torch
from torch import nn

seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(
            0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # dist = []
        # for i in range(batch_size):
        #     value = distmat[i][mask[i]]
        #     value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #     dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


if __name__ == '__main__':
    use_gpu = False
    center_loss = CenterLoss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor(
        [0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor(
            [0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    print(loss)
'''


'''
import cv2
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck

class MGN(nn.Module):
    def __init__(self, num_classes=1000, pool='avg', feats=1024):
        super(MGN, self).__init__()
        num_classes = num_classes

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(
            res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(
            res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(
            res_conv4), copy.deepcopy(res_p_conv5))

        if pool == 'max':
            pool2d = nn.MaxPool2d
        elif pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(
            2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        
        fg_p1 = fg_p1.view(x.size(0), -1)
        fg_p2 = fg_p2.view(x.size(0), -1)
        fg_p3 = fg_p3.view(x.size(0), -1)
        l_p1 = fg_p1
        l_p2 = fg_p2
        l_p3 = fg_p3

        l0_p2 = f0_p2
        l1_p2 = f1_p2
        l0_p3 = f0_p3
        l1_p3 = f1_p3
        l2_p3 = f2_p3

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2,
                             f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3


model = MGN()
a = cv2.imread(
    '/home/zjf/git/naic_person_reid_bak/data/test/query_a/00001655.png')
# print(a.shape)
a = cv2.resize(a, ( 128, 384))
x = torch.tensor(a).permute(2, 0, 1).float().unsqueeze(0)
x = torch.cat((x,x),0)

# print(x.shape)
feat = model(x)
print([x.shape for x in feat])
'''
