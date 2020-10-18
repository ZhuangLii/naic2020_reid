'''
compute naic 2020 score 
'''
import json
import os
from os.path import join
from collections import defaultdict
from sklearn.utils import shuffle
from tqdm import tqdm
from PIL import Image

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
        if not tmp_query_pid:
            raise ValueError('No matched quey!!!!')
        tmp_map = 0
        tmp_hit = 0
        tmp_top1 = 0
        for i,x in enumerate(predict_list):
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
    print('top1: {:.4f}'.format(total_top1 / total_len))
    print('map: {:.4f}'.format(total_map / total_len))
    final_score = 0.5 * total_top1 / total_len + 0.5 * total_map / total_len
    return final_score, bad_dict, good_dict

def eval_naic2020(result_path, gt_txt, vis, save_folder='/home/zjf/naic_code/vis_res'):
    with open(result_path, 'r') as f:
        res_dict = json.load(f)
    with open(gt_txt, 'r') as f:
        labels = f.readlines()
    img_pid_dict = defaultdict(str)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        img_pid_dict[img_name] = pid
    score, _, _ = cal_score(res_dict, img_pid_dict)
    print('score: {:.4f}'.format(score))
        
if __name__ == "__main__":
    from glob import glob
    result_root = './log/1229-a-1/*.json'
    result_path =  glob(result_root)[0]
    gt_txt = './mics/split_ids_for_train/g_q.txt'
    eval_naic2020(result_path, gt_txt, False)
