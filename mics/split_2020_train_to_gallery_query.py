import random
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from os.path import join
from tqdm import tqdm
RANDOM_STATE = 2020
random.seed(RANDOM_STATE)


def main(label_txt, dest_root):
    with open(label_txt, 'r') as f:
        labels = f.readlines()
    # labels, _ = train_test_split(
    #     labels, train_size=0.3, random_state=RANDOM_STATE, shuffle=True)
    pid_imgList = defaultdict(list)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        pid_imgList[pid].append(img_name)
    gallery_root = join(dest_root, 'gallery_a')
    query_root = join(dest_root, 'query_a')
    img_source_root = '/mnt/data/zhuangjunfei/NAIC_DATA_2020/train/images'
    for pid, img_name_list in tqdm(pid_imgList.items()):
        tmp_len = len(img_name_list)
        if tmp_len >=2 and tmp_len <= 14:
            a = 1
        elif tmp_len == 1:
            a = 0
        elif tmp_len > 14:
            a = tmp_len // 14
            # print(img_name_list, len(img_name_list))
        query_list = random.sample(img_name_list, a)
        gallery_list = list(set(img_name_list) - set(query_list))
        for q_name in query_list:
            os.system('ln -s ' + os.path.join(img_source_root, q_name) +
                      ' ' + os.path.join(query_root, q_name))
        for g_name in gallery_list:
            os.system('ln -s ' + os.path.join(img_source_root, g_name) +
                      ' ' + os.path.join(gallery_root, g_name))
        
if __name__ == "__main__":
    label_txt_root = '/mnt/data/zhuangjunfei/NAIC_DATA_2020/train/label.txt'
    dest_root = '/mnt/data/zhuangjunfei/NAIC_handcraft_2020'
    main(label_txt_root, dest_root)
