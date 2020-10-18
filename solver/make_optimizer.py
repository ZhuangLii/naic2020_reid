import torch
from .ranger import Ranger
from .SGD import SGD2
from torch.optim import SGD
import logging
def make_optimizer(cfg, model, center_criterion):
    logger = logging.getLogger("reid_baseline.train")
    use_gc = cfg.SOLVER.GRADCENTER
    gc_conv_only = False
    params = []
    momentum = 0.9
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                logger.info('Using two times learning rate for fc ')
        if "gap" in key:
            lr = cfg.SOLVER.BASE_LR * 10
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "momentum": momentum}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD_naive':
        optimizer = SGD(params, momentum=cfg.SOLVER.MOMENTUM)
        logger.info('use SGD_naive for optimizer ')
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = SGD2(model.parameters(), lr,
            momentum=cfg.SOLVER.MOMENTUM, use_gc=use_gc, gc_conv_only=gc_conv_only)
        logger.info('use SGD for optimizer ')
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Ranger':
        optimizer = Ranger(params, use_gc=use_gc, gc_conv_only=gc_conv_only)
        logger.info('use Ranger for optimizer ')
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    # else:
    #     optimizer = SGD2(model.parameters(), cfg.SOLVER.INIT_LR, momentum=cfg.SOLVER.MOMENTUM,
    #                      weight_decay=cfg.SOLVER.WEIGHT_DECAY, nesterov=True, use_gc=use_gc, gc_conv_only=gc_conv_only)
    #     logger.info('others lr scheduler using SGD for optimizer ')
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center
