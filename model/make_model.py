import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss, CurricularFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.resnet_ibn_b import resnet101_ibn_b
from .backbones.efficientnet import EfficientNet
from .backbones.densenet import densenet121, densenet169, densenet201, densenet161
from .backbones import osnet
from .backbones.sknet import SKNet101, SKNet26, SKNet50
from .backbones.CBAM import resnet101_cbam
from .backbones.non_local import Non_Local_101
from .backbones.inceptionv4 import inception
from .backbones.nasnet import ft_net_NAS
from .backbones.MGN import MGN_res50

import torch.nn.functional as F
from torch.nn.parameter import Parameter
class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        self.model_name = model_name
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.layernorm = cfg.MODEL.LAYERN0RM
        if self.layernorm:
            self.ln = nn.LayerNorm([cfg.MODEL.FEAT_SIZE])

        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.base = se_resnet101_ibn_a(last_stride)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_b':
            self.base = resnet101_ibn_b(last_stride)
            print('using resnet101_ibn_b as a backbone')
        elif model_name == 'efficientnet_b8':
            self.base = EfficientNet.from_pretrained('efficientnet-b8')
        elif model_name == 'efficientnet_b0':
            self.base = EfficientNet.from_pretrained('efficientnet-b0')
        elif model_name == 'efficientnet_b1':
            self.base = EfficientNet.from_pretrained('efficientnet-b1')
        elif model_name == 'efficientnet_b2':
            self.base = EfficientNet.from_pretrained('efficientnet-b2')
        elif model_name == 'efficientnet_b3':
            self.base = EfficientNet.from_pretrained('efficientnet-b3')
        elif model_name == 'efficientnet_b4':
            self.base = EfficientNet.from_pretrained('efficientnet-b4')
        elif model_name == 'efficientnet_b5':
            self.base = EfficientNet.from_pretrained('efficientnet-b5')
        elif model_name == 'efficientnet_b6':
            self.base = EfficientNet.from_pretrained('efficientnet-b6')
        elif model_name == 'efficientnet_b7':
            self.base = EfficientNet.from_pretrained('efficientnet-b7')
        elif model_name == 'densenet121':  # feat size 1024
            self.base = densenet121(pretrained=True)
        elif model_name == 'densenet161':
            self.base = densenet161(pretrained=True)
        elif model_name == 'densenet169':
            self.base = densenet169(pretrained=True)
        elif model_name == 'nasnet':
            self.base = ft_net_NAS() # feat size 4032
        elif model_name == 'osnet':
            self.base = osnet.build_osnet_backbone(cfg.MODEL.PRETRAIN_PATH)
        elif model_name == 'sknet':
            self.base = SKNet101()
        elif model_name == 'cbam':
            self.base = resnet101_cbam()
        elif model_name == 'non_local':
            self.base = Non_Local_101(last_stride)
        elif model_name == 'inceptionv4':  # feat size 1536
            self.base = inception()
        elif model_name == 'mgn':
            self.base = MGN_res50()
        else:
            print('unsupported backbone! but got {}'.format(model_name))
        self.in_planes = cfg.MODEL.FEAT_SIZE

        if pretrain_choice == 'imagenet' and cfg.MODEL.PRETRAIN_PATH and 'resnet101' in cfg.MODEL.NAME:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'curricularface':
            print('using {} loss with s:{}, m: {}'.format(
                self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CurricularFace(
                self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)

        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        if self.model_name != 'mgn':
            x = self.base(x)
            global_feat = self.gap(x)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            feat = self.bottleneck(global_feat)

            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)
            if self.training:
                try:
                    if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle', 'curricularface'):
                        cls_score = self.classifier(feat, label)
                    else:
                        cls_score = self.classifier(feat)
                    if not self.layernorm:
                        return cls_score, global_feat
                    else:
                        return cls_score, self.ln(global_feat)
                except:
                    pass
            else:
                if self.neck_feat == 'after':
                    # print("Test with feature after BN")
                    if not self.layernorm:
                        return feat
                    else:
                        self.ln(feat)
                else:
                    # print("Test with feature before BN")
                    if not self.layernorm:
                        return global_feat
                    else:
                        return self.ln(global_feat)
        else:
            pass
            # todo MGN

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
