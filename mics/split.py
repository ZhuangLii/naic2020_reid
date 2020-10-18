import random
import os
import shutil
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
RANDOM_STATE = 2020
random.seed(RANDOM_STATE)

def split_gallery_query(pid_imgList, train_ids, test_ids):
    train, gallery, query = defaultdict(list), defaultdict(list), defaultdict(list)
    for train_id in train_ids:
        train[train_id].extend(pid_imgList[train_id])
    for _id in test_ids:
        tmp_id_imgs = pid_imgList[_id]
        if len(tmp_id_imgs) < 2:
            gallery[_id].extend(tmp_id_imgs)
        else:
            q_img_num = len(tmp_id_imgs) // 14 + 1
            q_img_list = random.sample(tmp_id_imgs, q_img_num) 
            g_img_list = list(set(tmp_id_imgs).difference(set(q_img_list)))
            gallery[_id].extend(g_img_list)
            query[_id].extend(q_img_list)
    return train, gallery, query


def write_txt(output_path, pid_imgs_dict):
    with open(output_path, 'w') as f:
        for pid, img_list in pid_imgs_dict.items():
            for img in img_list:
                new_line = str(img) + ':' + str(pid) + '\n'
                f.write(new_line)

def makefolder(path):
    if not os.path.exists(path):
        os.mkdir(path) 


def copy_imgs(pid_img_dict, img_dest_root, source_img_root):
    for _, img_list in tqdm(pid_img_dict.items()):
        for img_name in img_list:
            shutil.copy(os.path.join(source_img_root, img_name),
                    os.path.join(img_dest_root, img_name))


def main(label_txt, dest_root, img_source_root, split_num=4):
    with open(label_txt, 'r') as f:
        labels = f.readlines()
    pid_imgList = defaultdict(list)
    for label in labels:
        img_name, pid = label.rstrip().split(':')
        pid_imgList[pid].append(img_name)
    pid_list = list(pid_imgList.keys())
    kf = KFold(n_splits=split_num, shuffle=True)
    split_idx = 0
    for train_idx, test_idx in kf.split(pid_list):
        # split
        train_pid_idx = [pid_list[x] for x in train_idx]
        test_pid_idx = [pid_list[x] for x in test_idx]
        train_dict, gallery_dict, query_dict = split_gallery_query(pid_imgList, train_pid_idx, test_pid_idx)
        # make folder
        tmp_dest_root = os.path.join(dest_root, 'split' + str(split_idx))
        makefolder(tmp_dest_root)
        tmp_dest_train_root = os.path.join(tmp_dest_root, 'train')
        makefolder(tmp_dest_train_root)
        tmp_dest_test_root = os.path.join(tmp_dest_root, 'test')
        makefolder(tmp_dest_test_root)
        tmp_dest_gallery_root = os.path.join(tmp_dest_test_root, 'gallery_a')
        makefolder(tmp_dest_gallery_root)
        tmp_dest_query_root = os.path.join(tmp_dest_test_root, 'query_a')
        makefolder(tmp_dest_query_root)
        tmp_dest_train_img_root = os.path.join(tmp_dest_train_root, 'images')
        makefolder(tmp_dest_train_img_root)
        # write label
        write_txt(tmp_dest_train_root + '/label.txt', train_dict)
        # copy images
        copy_imgs(train_dict, tmp_dest_train_img_root, img_source_root)
        copy_imgs(gallery_dict, tmp_dest_gallery_root, img_source_root)
        copy_imgs(query_dict, tmp_dest_query_root, img_source_root)
        split_idx += 1


if __name__ == "__main__":
    
    dest_root = '/mnt/nfs-internstorage/train_data/naic_4split'
    label_txt = '/mnt/nfs-internstorage/train_data/naic2020/train/label.txt'
    img_source_root = '/mnt/nfs-internstorage/train_data/naic2020/train/images'
    main(label_txt, dest_root, img_source_root)
