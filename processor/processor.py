import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
import shutil
from torch.nn.parallel import DistributedDataParallel
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_Pseudo
import json
import datetime
from collections import defaultdict
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, score, feat, y_a, y_b, lam):
    loss_a = criterion(score, feat, y_a)
    loss_b = criterion(score, feat, y_b)
    total_a_loss, id_a_loss, tri_a_loss, cen_a_loss = loss_a
    total_b_loss, id_b_loss, tri_b_loss, cen_b_loss = loss_b
    return (lam*total_a_loss + (1-lam)*total_b_loss,
            lam*id_a_loss+(1-lam)*id_b_loss,
            lam*tri_a_loss + (1-lam)*tri_b_loss,
            lam*cen_a_loss + (1-lam)*cen_b_loss)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query,
             writer):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    tmp_input_data = torch.rand(
        (10, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]))
    writer.add_graph(model, (tmp_input_data,))

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            model = model.cuda()
        else:
            if cfg.SOLVER.FP16:
                model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    cen_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    lr_meter = AverageMeter()
    if cfg.SOLVER.SWA:
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        id_loss_meter.reset()
        tri_loss_meter.reset()
        cen_loss_meter.reset()
        lr_meter.reset()
        model.train()
        if cfg.SOLVER.GRADUAL_UNLOCK:
            model.base.gradual_unlock(cfg.SOLVER.MAX_EPOCHS, epoch)
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.DATASETS.MIXUP:
                img, target_a, target_b, lam = mixup_data(img, target)
            score, feat = model(img, target)

            if cfg.DATASETS.MIXUP:
                all_loss = mixup_criterion(
                    loss_fn, score, feat, target_a, target_b, lam)
            else:
                all_loss = loss_fn(score, feat, target)
            loss, id_loss, tri_loss, cen_loss = all_loss
            if cfg.SOLVER.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            loss_meter.update(loss.item(), img.shape[0])
            id_loss_meter.update(id_loss.item(), img.shape[0])
            if torch.is_tensor(tri_loss): 
                tri_loss_meter.update(tri_loss.item(), img.shape[0])
            else:
                tri_loss_meter.update(tri_loss, 1)
            if torch.is_tensor(cen_loss):
                cen_loss_meter.update(cen_loss.item(), img.shape[0])
            else:
                cen_loss_meter.update(cen_loss, 1)
            acc = (score.max(1)[1] == target).float().mean()
            acc_meter.update(acc, 1)
            lr_meter.update(scheduler.get_last_lr()[0])

            writer.add_scalar(
                'data/total_loss',  loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar(
                'data/id_loss',  id_loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar( 
                'data/tri_loss',  tri_loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar(
                'data/cen_loss',  cen_loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar(
                'data/learning_rate', lr_meter.avg, (epoch-1)*len(train_loader) + n_iter)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_last_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            src_path = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
            torch.save(model.state_dict(), src_path)
            try:
                dest_root = os.path.join('/mnt/nfs-internstorage/user/zjf/NAIC2020/models', cfg.SAVE_FLAG)
                if not os.path.exists(dest_root):
                    os.mkdir(dest_root)
                dst_path = os.path.join(dest_root,
                    cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                shutil.copy(src_path, dst_path)
            except:
                print('No bak models...')
                pass
        if cfg.SOLVER.SWA and epoch in cfg.SOLVER.SWA_START:
            swa_model.update_parameters(model)
            logger.info('swa combine the {} epoch model'.format(epoch))
    if cfg.SOLVER.SWA:
        try: 
            swa_model.cpu()
            torch.optim.swa_utils.update_bn(train_loader, swa_model)
            swa_model.cuda()
            src_path = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_swa.pth')
            torch.save(swa_model.state_dict(), src_path)
            logger.info('swa model is successfuly saved.')
        except:
            logger.info('swa model save failed.')

def do_inference(cfg,
                 model,
                 val_loader_green,
                val_loader_normal,
                 num_query_green,
                 num_query_normal):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23436', rank=0, world_size=1)
            # model = DistributedDataParallel(model)
            # model = model.cuda()
            # model = nn.parallel.DistributedDataParallel(model)
            model = nn.DataParallel(model)
            model = model.cuda()
        model.to(device)

    model.eval()
    val_loader = [val_loader_green, val_loader_normal]
    for index, loader in enumerate(val_loader):
        if index == 0:
            subfix = '1'
            reranking_parameter = cfg.TEST.RE_RANKING_PARAMETER_GREEN
            evaluator = R1_mAP(num_query_green, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                               reranking=cfg.TEST.RE_RANKING)
        else:
            subfix = '2'
            reranking_parameter = cfg.TEST.RE_RANKING_PARAMETER_NORMAL
            evaluator = R1_mAP(num_query_normal, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                               reranking=cfg.TEST.RE_RANKING)

        evaluator.reset()
        DISTMAT_PATH = os.path.join(cfg.OUTPUT_DIR, "distmat_{}.npy".format(subfix))
        QUERY_PATH = os.path.join(cfg.OUTPUT_DIR, "query_path_{}.npy".format(subfix))
        GALLERY_PATH = os.path.join(cfg.OUTPUT_DIR, "gallery_path_{}.npy".format(subfix))
        
        # feat_imagepath_list = [] 
        # FEATS_IMAGEPATH_LIST_PATH = os.path.join(
        #     "/home/zjf/naic_code/data/feats_imagepath_list", "curricularface_m05s40_{}.npy".format(subfix))
        
        for n_iter, (img, pid, camid, imgpath) in enumerate(loader):
            with torch.no_grad():
                img = img.to(device)
                if cfg.TEST.FLIP_FEATS == 'on':
                    feat = torch.FloatTensor(
                        img.size(0), cfg.MODEL.FEAT_SIZE).zero_().cuda()
                    for i in range(2):
                        if i == 1:
                            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                            img = img.index_select(3, inv_idx)
                        f = model(img)
                        feat = feat + f
                else:
                    feat = model(img)

                evaluator.update((feat, imgpath))
        #         feat_imagepath_list.append((feat, imgpath))
        # np.save(FEATS_IMAGEPATH_LIST_PATH, feat_imagepath_list)

        data, distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)
        np.save(DISTMAT_PATH, distmat)
        np.save(QUERY_PATH, img_name_q)
        np.save(GALLERY_PATH, img_name_g)

        if index == 0:
            data_1 = data

    data_all = {**data_1, **data}
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(cfg.OUTPUT_DIR, 'result_{}.json'.format(nowTime)), 'w',encoding='utf-8') as fp:
        json.dump(data_all, fp)


def do_inference_Pseudo(cfg,
                 model,
                val_loader,
                num_query
                 ):
    device = "cuda"

    evaluator = R1_mAP_Pseudo(num_query, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # torch.distributed.init_process_group(backend="nccl")
            # model = DistributedDataParallel(model)
            # model = model.cuda()
            # model = nn.parallel.DistributedDataParallel(model)
            model = nn.DataParallel(model)
            model = model.cuda()
        model.to(device)

    reranking_parameter = [14, 4, 0.4]

    model.eval()
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(
                    img.size(0), cfg.MODEL.FEAT_SIZE).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, imgpath))

    distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)
    return distmat, img_name_q, img_name_g


def cal_score(res_dict, img_pid_dict):
    '''
    res_dict: {'query_name': [pred1_name, pred2_name ...]}
    '''
    total_len = len(res_dict)
    total_top1 = 0
    total_map = 0
    bad_dict = {}
    good_dict = {}
    for query, predict_list in res_dict.items():
        tmp_query_pid = img_pid_dict[query]
        if tmp_query_pid == '':
            raise ValueError("tmp_query_pid == null ")
        tmp_map = 0
        tmp_hit = 0
        tmp_top1 = 0
        for i, x in enumerate(predict_list):
            tmp_len = i + 1
            if img_pid_dict[x] == tmp_query_pid:
                tmp_hit += 1
                tmp_map += tmp_hit / tmp_len
        if not tmp_hit == 0:
            tmp_map = tmp_map / tmp_hit
        else:
            tmp_map = 0
        if img_pid_dict[predict_list[0]] == tmp_query_pid:
            tmp_top1 = 1
        else:
            tmp_top1 = 0
        total_top1 += tmp_top1
        total_map += tmp_map
        tmp_score = (tmp_top1 + tmp_map) / 2
        if tmp_score < 0.3:
            bad_dict[query] = predict_list
        if tmp_score > 0.7:
            good_dict[query] = predict_list
    final_score = 0.5 * total_top1 / total_len + 0.5 * total_map / total_len
    return final_score, bad_dict, good_dict


def do_train_val(cfg,
            model,
            center_criterion,
            train_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_fn,
            writer,
            val_loader_green,
            val_loader_normal,
            num_query_green,
            num_query_normal):

    with open(cfg.EVAL_GT_PATH, 'r') as f:
        labels = f.readlines()
    img_pid_dict = defaultdict(str)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        img_pid_dict[img_name] = pid
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    tmp_input_data = torch.rand(
        (10, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]))
    writer.add_graph(model, (tmp_input_data,))

    logger = logging.getLogger("reid_baseline.train.val")
    logger.info('start training')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            model = model.cuda()
        else:
            if cfg.SOLVER.FP16:
                model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    cen_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    lr_meter = AverageMeter()
    # train
    val_loader = [val_loader_green, val_loader_normal]
    if cfg.SOLVER.SWA:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        id_loss_meter.reset()
        tri_loss_meter.reset()
        cen_loss_meter.reset()
        lr_meter.reset()
        model.train()
        if cfg.SOLVER.GRADUAL_UNLOCK:
            model.base.gradual_unlock(cfg.SOLVER.MAX_EPOCHS, epoch)
            # for model.base: TODO record learning rate and required 
            #     logger.info()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.DATASETS.MIXUP:
                img, target_a, target_b, lam = mixup_data(img, target)
            score, feat = model(img, target)

            if cfg.DATASETS.MIXUP:
                all_loss = mixup_criterion(
                    loss_fn, score, feat, target_a, target_b, lam)
            else:
                all_loss = loss_fn(score, feat, target)
            loss, id_loss, tri_loss, cen_loss = all_loss
            if cfg.SOLVER.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            loss_meter.update(loss.item(), img.shape[0])
            id_loss_meter.update(id_loss.item(), img.shape[0])
            if torch.is_tensor(tri_loss): 
                tri_loss_meter.update(tri_loss.item(), img.shape[0])
            else:
                tri_loss_meter.update(tri_loss, 1)
            if torch.is_tensor(cen_loss):
                cen_loss_meter.update(cen_loss.item(), img.shape[0])
            else:
                cen_loss_meter.update(cen_loss, 1)
            acc = (score.max(1)[1] == target).float().mean()
            acc_meter.update(acc, 1)
            lr_meter.update(scheduler.get_last_lr()[0])

            writer.add_scalar(
                'data/total_loss',  loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar(
                'data/id_loss',  id_loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar( 
                'data/tri_loss',  tri_loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar(
                'data/cen_loss',  cen_loss_meter.avg, (epoch-1)*len(train_loader) + n_iter)
            writer.add_scalar(
                'data/learning_rate', lr_meter.avg, (epoch-1)*len(train_loader) + n_iter)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_last_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            src_path = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
            torch.save(model.state_dict(), src_path)
            try:
                dest_root = os.path.join('/mnt/nfs-internstorage/user/zjf/NAIC2020/models', cfg.SAVE_FLAG)
                if not os.path.exists(dest_root):
                    os.mkdir(dest_root)
                dst_path = os.path.join(dest_root,
                    cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                shutil.copy(src_path, dst_path)
            except:
                print('No bak models...')
                pass
        if cfg.SOLVER.SWA and epoch in cfg.SOLVER.SWA_START:
            swa_model.update_parameters(model)
            logger.info('swa combine the {} epoch model'.format(epoch))

        if epoch >= cfg.EVAL_START_EPOCH:
            model.eval()
            res_dict = eval_fun(cfg, model, val_loader, num_query_green, num_query_normal, device)
            score,_, _ = cal_score(res_dict, img_pid_dict)
            writer.add_scalar('data/val_score', score, epoch)
            logger.info('{} epoch eval score {:.4f}'.format(epoch, score))

    if cfg.SOLVER.SWA:
        try:
            swa_model.cpu()
            torch.optim.swa_utils.update_bn(train_loader, swa_model)
            swa_model.cuda()
            src_path = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_swa.pth')
            torch.save(swa_model.state_dict(), src_path)
            logger.info('swa model is successfuly saved.')
        except:
            logger.info('swa model save failed.')
    
    if cfg.SOLVER.SWA and epochs in cfg.SOLVER.SWA_START:
        try:
            swa_model.eval()
            res_dict = eval_fun(cfg, swa_model, val_loader,
                                num_query_green, num_query_normal, device)
            score, _, _ = cal_score(res_dict, img_pid_dict)
            logger.info('swa model success in evaluating.')
            logger.info('swa model eval score {:.4f}'.format(score))
        except:
            logger.info('swa model fail in evaluating.')


def eval_fun(cfg, model, val_loader, num_query_green, num_query_normal, device):
    for index, loader in enumerate(val_loader):
        if index == 0:
            reranking_parameter = cfg.TEST.RE_RANKING_PARAMETER_GREEN
            evaluator = R1_mAP(num_query_green, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING)
        else:
            reranking_parameter = cfg.TEST.RE_RANKING_PARAMETER_NORMAL
            evaluator = R1_mAP(num_query_normal, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING)
        evaluator.reset()
        for n_iter, (img, pid, camid, imgpath) in enumerate(loader):
            with torch.no_grad():
                img = img.to(device)
                if cfg.TEST.FLIP_FEATS == 'on':
                    feat = torch.FloatTensor(
                        img.size(0), cfg.MODEL.FEAT_SIZE).zero_().cuda()
                    for i in range(2):
                        if i == 1:
                            inv_idx = torch.arange(
                                img.size(3) - 1, -1, -1).long().cuda()
                            img = img.index_select(3, inv_idx)
                        f = model(img)
                        feat = feat + f
                else:
                    feat = model(img)

                evaluator.update((feat, imgpath))
        data, distmat, img_name_q, img_name_g = evaluator.compute(
            reranking_parameter)
        if index == 0:
            data_1 = data
    res_dict = {**data_1, **data}
    return res_dict
