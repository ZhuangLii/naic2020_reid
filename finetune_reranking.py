
import numpy as np
from os.path import join
from utils.metrics import R1_mAP, R1_mAP_Pseudo
from config import cfg
import argparse
from datasets import make_dataloader
import datetime
import json
import os
import random
import logging
from collections import defaultdict
from sko.GA import GA
from sko.PSO import PSO

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
    final_score = 0.5 * total_top1 / total_len + 0.5 * total_map / total_len
    return final_score


def get_logger(logger_path):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logger_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def tune_paramters(cfg, img_pid_dict, feat_imgpath_list, nums_list, logger, finetunemethod, max_iters):
    def opt_fun(paras):
        reranking_para_list = [(int(round(paras[0])), 
                                int(round(paras[1])), 
                                paras[2]),
                               (int(round(paras[3])),
                               int(round(paras[4])), 
                                paras[5])]
        
        for index in range(2):
            num_tmp = nums_list[index]
            evaluator = R1_mAP(num_tmp, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=True)
            evaluator.reset()
            feat_imgpath = feat_imgpath_list[index]
            for item in feat_imgpath:
                evaluator.update(item)
            data, distmat, img_name_q, img_name_g = evaluator.compute(
                reranking_para_list[index])
            if index == 0:
                data_1 = data
        res_dict = {**data_1, **data}
        score = cal_score(res_dict, img_pid_dict)
        return -1 * score

    if finetunemethod == 'ga':
        ga = GA(func=opt_fun, n_dim=6, size_pop=4, max_iter=8,
                lb=[5, 1, 0.1, 5, 1, 0.1], ub=[20, 10, 0.9, 20, 10, 0.9], precision=0.1)
        for _ in range(max_iters):
            tmp_best_paras, score = ga.run(2)
            logger.info(
                "Parameter1: {:.4f}, {:.4f}, {:.4f} Parameter2: {:.4f}, {:.4f}, {:.4f}, score: {:.4f} ".format(float(tmp_best_paras[0]), float(tmp_best_paras[1]), float(tmp_best_paras[2]), float(tmp_best_paras[3]), float(tmp_best_paras[4]), float(tmp_best_paras[5]), float(score)))
        
    elif finetunemethod == 'pso':
        pso = PSO(func=opt_fun, dim=6, pop=5, max_iter=5, lb=[5, 1, 0.1, 5, 1, 0.1], ub=[20, 10, 0.9, 20, 10, 0.9], w=0.2, c1=0.1, c2=0.1)
        for _ in range(max_iters):
            pso.run(3)
            logger.info(
                "Parameter1: {:.4f}, {:.4f}, {:.4f} Parameter2: {:.4f}, {:.4f}, {:.4f}, score: {:.4f} ".format(pso.gbest_x[0], pso.gbest_x[1], pso.gbest_x[2], pso.gbest_x[3], pso.gbest_x[4], pso.gbest_x[5], pso.gbest_y))


def main():
    finetunemethod = 'ga'
    max_iters = 5
    label_txt = '/home/zjf/naic_code/data/naic2020_evaldataset/test/g_q.txt'
    cfg.merge_from_file('./configs/naic_round2_model_a_local.yml')
    g_feat_imgpath = '/home/zjf/naic_code/data/feats_imagepath_list/saved_feat_imagepath_list_1.npy'
    n_feat_imgpath = '/home/zjf/naic_code/data/feats_imagepath_list/saved_feat_imagepath_list_2.npy'
    logger = get_logger("../log/reranking_finetune.log")

    with open(label_txt, 'r') as f:
        labels = f.readlines()
    img_pid_dict = defaultdict(str)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        img_pid_dict[img_name] = pid
    feat_imgpath_path_list = [g_feat_imgpath, n_feat_imgpath]
    feat_imgpath_list = []
    for item in feat_imgpath_path_list:
        feat_imgpath = np.load(item, allow_pickle=True)
        feat_imgpath = feat_imgpath.tolist()
        feat_imgpath_list.append(feat_imgpath)
    _, val_loader_green, val_loader_normal, num_query_green, num_query_normal, _ = make_dataloader(
        cfg)
    val_loaders = [val_loader_green, val_loader_normal]
    nums_list = [num_query_green, num_query_normal]
    tune_paramters(cfg, img_pid_dict, feat_imgpath_list,
                   nums_list, logger, finetunemethod, max_iters)


if __name__ == "__main__":
    main()
