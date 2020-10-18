import os
from torch.backends import cudnn
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor.processor import do_train_val
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, val_loader_green, val_loader_normal, num_query_green, num_query_normal, num_classes = make_dataloader(cfg)

    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        model = make_model(cfg, num_class=num_classes)
        model.load_param_finetune(cfg.MODEL.PRETRAIN_PATH)
        print('Loading pretrained model for finetuning......')
    else:
        model = make_model(cfg, num_class=num_classes)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    
    if cfg.SOLVER.TYPE == 'warmup':
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    elif cfg.SOLVER.TYPE == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA)
    elif cfg.SOLVER.TYPE == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8685)
    elif cfg.SOLVER.TYPE == 'cosineann':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.SOLVER.T_MAX, eta_min=cfg.SOLVER.ETA_MIN, last_epoch=-1)
    else:
        raise ValueError('invaild cfg.SOLVER.TYPE parameters')
    if cfg.SOLVER.SWA:
        logger.info("SWA is used for combine trained models...")

    writer = SummaryWriter(cfg.TBOUTPUT_DIR)
    do_train_val(cfg,
            model,
            center_criterion,
            train_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            writer,
            val_loader_green,
            val_loader_normal,
            num_query_green,
            num_query_normal)
