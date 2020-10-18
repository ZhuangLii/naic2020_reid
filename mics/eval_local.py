'''
compute naic 2020 score 
'''
import json
import os
import random
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
    print('top1: {:.4f}'.format(total_top1 / total_len))
    print('map: {:.4f}'.format(total_map / total_len))
    return final_score, bad_dict, good_dict


def visual(bad_dict, save_folder, num=10e9, flag='bad'):
    count = 0
    for query, pred_list in tqdm(bad_dict.items()):
        pic_list = [query] + pred_list[:9]

        images = [Image.open(os.path.join(
            '/mnt/data/zhuangjunfei/NAIC_DATA_2020/train/images', x)).resize((128, 256)) for x in pic_list]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(os.path.join(
            save_folder, flag + '_{}.jpg'.format(count)))
        count += 1
        if count > num:
            print('draw done!')
            break


def visual_groundtruth():
    dest_root = '/home/zjf/naic_code/vis_res'
    label_txt = '/mnt/data/zhuangjunfei/NAIC_DATA_2020/train/label.txt'
    save_folder = '/home/zjf/naic_code/vis_res'
    with open(label_txt, 'r') as f:
        labels = f.readlines()
    pid_imgList = defaultdict(list)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        pid_imgList[pid].append(img_name)
    count = 0
    for k, v in tqdm(pid_imgList.items()):
        tmp_img_list = shuffle(v)
        tmp_img_list = tmp_img_list[:10]
        images = [Image.open(os.path.join(
            '/mnt/data/zhuangjunfei/NAIC_DATA_2020/train/images', x)).resize((128, 256)) for x in tmp_img_list]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(os.path.join(
            save_folder, 'gt_{}.jpg'.format(count)))
        count += 1


def eval_naic2020(result_path, gt_txt, vis, save_folder='/home/zjf/naic_code/vis_res'):
    with open(result_path, 'r') as f:
        res_dict = json.load(f)
    with open(gt_txt, 'r') as f:
        labels = f.readlines()
    img_pid_dict = defaultdict(str)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        img_pid_dict[img_name] = pid
    score, bad_dict, good_dict = cal_score(res_dict, img_pid_dict)
    print('score: {:.4f}'.format(score))
    # with open('a_auto.json', 'w') as f:
    #    json.dump(bad_dict, f)
    if vis:
        visual(bad_dict, save_folder)
        visual(good_dict, save_folder, flag='good')


if __name__ == "__main__":
    from glob import glob
    import argparse
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--folder", "-f", default="")
    parser.add_argument("--vis", "-v", default=False)
    parser.add_argument("--vis_gt", "-vg", default=False)
    args = parser.parse_args()
    result_root = '../log/19_20_a_auto/*.json'
    if len(glob(result_root))>1:
        raise ValueError('too many json results')
    result_path = glob(result_root)[0]
    gt_txt = './mics/split_ids_for_train/g_q.txt'
    # result_path = '/mnt/nfs-internstorage/user/zjf/NAIC2020/ensemble_results/new_val_a_auto_curricularface_m05s40new_val_a_auto_curricularface_m03s20new_val_a_auto_center_v2/submit_final.json'
    # gt_txt = '../split_ids_for_train/g_q.txt'
    eval_naic2020(result_path, gt_txt, False)
