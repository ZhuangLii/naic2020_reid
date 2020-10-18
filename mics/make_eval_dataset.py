import random
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from os.path import join

RANDOM_STATE = 2009
random.seed(RANDOM_STATE)


def jugde_normal_or_green(img_path):
    img = cv2.imread(img_path)
    per_image_Bmean = np.mean(img[:, :, 0])
    per_image_Gmean = np.mean(img[:, :, 1])
    per_image_Rmean = np.mean(img[:, :, 2])
    if per_image_Bmean > 65 and per_image_Gmean > 65 and per_image_Rmean > 65:
        return 'green'
    else:
        return 'normal'


def main(label_txt, img_source_root, dest_root, train19root, test_20_path, query_normal_num):
    gallery_20_path = join(test_20_path, 'gallery')
    query_20_path = join(test_20_path, 'query')
    print('calculate naic 2020 green and normal images..')
    tmp_imgs = os.listdir(gallery_20_path)
    tmp_g_counter, tmp_n_counter = 0, 0
    for x in tmp_imgs:
        tmp_res = jugde_normal_or_green(join(gallery_20_path, x))
        if tmp_res == 'normal':
            tmp_n_counter += 1
        elif tmp_res == 'green':
            tmp_g_counter += 1
    print("{} total gallery images".format(len(tmp_imgs)))
    print("{} total green gallery images".format(tmp_g_counter))
    print("{} total normal gallery images".format(tmp_n_counter))
    print('----'*8)
    tmp_imgs = os.listdir(query_20_path)
    tmp_g_counter, tmp_n_counter = 0, 0
    for x in tmp_imgs:
        tmp_res = jugde_normal_or_green(join(query_20_path, x))
        if tmp_res == 'normal':
            tmp_n_counter += 1
        elif tmp_res == 'green':
            tmp_g_counter += 1
    print("{} total query images".format(len(tmp_imgs)))
    print("{} total green query images".format(tmp_g_counter))
    print("{} total normal query images".format(tmp_n_counter))
    print('----'*8)

    query_green_num = query_normal_num * 0.86667
    gallery_green_num = query_normal_num * 2.73333
    gallery_normal_num = query_normal_num * 23.83005
    g_dest_root = join(dest_root, 'test/gallery_a')
    q_dest_root = join(dest_root, 'test/query_a')
    train_dest_root = join(dest_root, 'train/images')
    train_label_path = join(dest_root, 'train/label.txt')
    test_label_path = join(dest_root, 'test/g_q.txt')
    print('count green and normal image...')

    with open(label_txt, 'r') as f:
        labels = f.readlines()
    pid_imgList_green = defaultdict(list)
    pid_imgList_normal = defaultdict(list)
    for label in tqdm(labels):
        img_name, pid = label.rstrip().split(':')
        if pid not in pid_imgList_green and pid not in pid_imgList_normal:
            img_path = join(img_source_root, img_name)
            g_or_n = jugde_normal_or_green(img_path)
            if g_or_n == "green":
                pid_imgList_green[pid].append(img_name)
            elif g_or_n == "normal":
                pid_imgList_normal[pid].append(img_name)
            else:
                raise ValueError("{} not belong to green or normal".format(img_name))
        elif pid in pid_imgList_green:
            pid_imgList_green[pid].append(img_name)
        elif pid in pid_imgList_normal:
            pid_imgList_normal[pid].append(img_name)
        else:
            raise ValueError("{} not belong to green or normal".format(img_name))
    pid_green_list = list(pid_imgList_green.keys())
    random.shuffle(pid_green_list)
    pid_normal_list = list(pid_imgList_normal.keys())
    random.shuffle(pid_normal_list)
    green_img_list = []
    for x in pid_green_list:
        green_img_list.extend(pid_imgList_green[x])
    normal_img_list = []
    for x in pid_normal_list:
        normal_img_list.extend(pid_imgList_normal[x])
    print("2020 naic train has {} all images, {} green images, {} normal images".format(len(labels),len(green_img_list),len(normal_img_list)))
    print('----'*8)


    # establish query normal
    print("establish query normal...")
    n_q_counter = 0
    n_g_counter = 0
    for tmp_pid in tqdm(pid_normal_list):
        tmp_len = len(pid_imgList_normal[tmp_pid])
        if tmp_len <= 1:
            continue
        else:
            tmp_q_num = tmp_len // 14 + 1
            tmp_q_img_list = random.sample(pid_imgList_normal[tmp_pid], tmp_q_num) 
            tmp_g_img_list = list(set(pid_imgList_normal[tmp_pid]).difference(set(tmp_q_img_list)))
            assert len(tmp_q_img_list) > 0 and len(tmp_g_img_list) > 0
            assert all([x in pid_imgList_normal[tmp_pid] for x in tmp_q_img_list])
            assert all([x in pid_imgList_normal[tmp_pid] for x in tmp_g_img_list])
            assert len(set(tmp_q_img_list) & set(tmp_g_img_list)) == 0
            with open(test_label_path, 'a') as f:
                for x in tmp_q_img_list:
                    shutil.copy(join(img_source_root, x), join(q_dest_root, x))
                    tmp_label = x.rstrip() + ':' + tmp_pid + '\n'
                    f.write(tmp_label)
                for x in tmp_g_img_list:
                    shutil.copy(join(img_source_root, x), join(g_dest_root, x))
                    tmp_label = x.rstrip() + ':' + tmp_pid + '\n'
                    f.write(tmp_label)
            n_q_counter += len(tmp_q_img_list)
            n_g_counter += len(tmp_g_img_list)
            pid_imgList_normal[tmp_pid] = []
        if n_q_counter > query_normal_num:
            break
    print("In our query: {} normal images".format(n_q_counter))
    print("In our gallery: {} normal images".format(n_g_counter))
    tmp_num = sum([len(pid_imgList_normal[x]) for x in pid_normal_list])
    print("2020 left normal images {}".format(tmp_num))
    print('----'*8)

    # move all left normal image to gallery
    '''
    print("establish gallery normal...")
    for tmp_pid in tqdm(list(pid_imgList_normal.keys())):
        if n_g_counter > gallery_normal_num:
            break
        with open(test_label_path, 'a') as f:
            for x in pid_imgList_normal[tmp_pid]:
                assert x in normal_img_list
                shutil.copy(join(img_source_root, x), join(g_dest_root, x))
                tmp_label = x.rstrip() + ':' + tmp_pid + '\n'
                f.write(tmp_label)
        n_g_counter += len(pid_imgList_normal[tmp_pid])
        pid_imgList_normal[tmp_pid] = []
    '''

    # establish query green
    print("establish query green...")
    g_q_counter = 0
    g_g_counter = 0
    for tmp_pid in tqdm(pid_green_list):
        tmp_len = len(pid_imgList_green[tmp_pid])
        if tmp_len <= 1:
            continue
        else:
            tmp_g_num = tmp_len // 14 + 1
            tmp_q_img_list = random.sample(pid_imgList_green[tmp_pid], tmp_g_num) 
            tmp_g_img_list = list(set(pid_imgList_green[tmp_pid]).difference(set(tmp_q_img_list)))
            assert len(tmp_q_img_list) > 0 and len(tmp_g_img_list) > 0
            assert all([x in pid_imgList_green[tmp_pid] for x in tmp_q_img_list])
            assert all([x in pid_imgList_green[tmp_pid] for x in tmp_g_img_list])
            assert len(set(tmp_q_img_list) & set(tmp_g_img_list)) == 0
            with open(test_label_path, 'a') as f:
                for x in tmp_q_img_list:
                    shutil.copy(join(img_source_root, x), join(q_dest_root, x))
                    tmp_label = x.rstrip() + ':' + tmp_pid + '\n'
                    f.write(tmp_label)
                for x in tmp_g_img_list:
                    shutil.copy(join(img_source_root, x), join(g_dest_root, x))
                    tmp_label = x.rstrip() + ':' + tmp_pid + '\n'
                    f.write(tmp_label)
            g_q_counter += len(tmp_q_img_list)
            g_g_counter += len(tmp_g_img_list)
            pid_imgList_green[tmp_pid] = []
        if g_q_counter > query_green_num:
            break
    print("In our query: {} green images".format(g_q_counter))
    print("In our gallery: {} green images".format(g_g_counter))
    tmp_num = sum([len(pid_imgList_green[x]) for x in pid_green_list])
    print("2020 left green images {}".format(tmp_num))
    print('----'*8)
    # use left green image to fit gallery
    print("use left green image to fit gallery...")
    tmp_counter = 0
    for tmp_pid in tqdm(list(pid_imgList_green.keys())):
        if g_g_counter > gallery_green_num:
            break
        with open(test_label_path, 'a') as f:
            for x in pid_imgList_green[tmp_pid]:
                assert x in green_img_list
                shutil.copy(join(img_source_root, x), join(g_dest_root, x))
                tmp_label = x.rstrip() + ':' + tmp_pid + '\n'
                f.write(tmp_label)
        g_g_counter += len(pid_imgList_green[tmp_pid])
        tmp_counter += len(pid_imgList_green[tmp_pid])
        pid_imgList_green[tmp_pid] = []
    print("copy {} green images to our gallery".format(tmp_counter))
    tmp_num = sum([len(pid_imgList_green[x]) for x in pid_green_list])
    print("2020 left green images {}".format(tmp_num))
    print('----'*8)

    # check
    print("check...")
    g_imgs = os.listdir(g_dest_root)
    q_imgs = os.listdir(q_dest_root)
    q_g_imgs = g_imgs + q_imgs
    if len(set(g_imgs) & set(q_imgs)) == 0:
        print("ok")
    else:
        print("query have same image with gallery", list(set(g_imgs) & set(q_imgs)))
    with open(label_txt, 'r') as f_ori:
        ori_labels = f_ori.readlines()
        with open(test_label_path, 'r') as f:
            for x in f.readlines():
                assert x.rstrip().split(':')[0] in q_g_imgs
                assert x in ori_labels



    # establish train set
    print('establish train set...')
    train_label = 0
    tmp_counter_g = 0
    tmp_counter_n = 0
    with open(train_label_path, 'a') as f:
        for k, v in pid_imgList_green.items():
            if len(v) == 0:
                continue
            else:
                for tmp_img_name in v:
                    shutil.copy(join(img_source_root, tmp_img_name), join(train_dest_root, tmp_img_name))
                    tmp_label = tmp_img_name.rstrip() + ':' + str(train_label) + '\n'
                    f.write(tmp_label)
                    tmp_counter_g += 1
                train_label += 1
    
    with open(train_label_path, 'a') as f:
        for k, v in pid_imgList_normal.items():
            if len(v) == 0:
                continue
            else:
                for tmp_img_name in v:
                    shutil.copy(join(img_source_root, tmp_img_name), join(train_dest_root, tmp_img_name))
                    tmp_label = tmp_img_name.rstrip() + ':' + str(train_label) + '\n'
                    f.write(tmp_label)
                    tmp_counter_n += 1
                train_label += 1
    print("copy {} green images to train set".format(tmp_counter_g))
    print("copy {} normal images to train set".format(tmp_counter_n))
    print("copy {} total images to train set, tmp label {}".format(tmp_counter_g + tmp_counter_n, train_label))
    print('----'*8)
    '''
    print("check...")
    train_imgs = os.listdir(train_dest_root)
    with open(label_txt, 'r') as f_ori:
        ori_labels = f_ori.readlines()
        with open(test_label_path, 'r') as f_test:
            test_labels = f_test.readlines()
            with open(train_label_path, 'r') as f_train:
                train_labels = f_train.readlines()
                for x in train_labels:
                    assert x in ori_labels
                    assert x not in test_labels
                assert len(train_labels) == len(train_imgs)
            assert len(test_labels) + len(train_labels) == len(ori_labels)
    '''

    print('append 2019 imgs to train')
    train19_imgroot = join(train19root, 'images')
    tmp_19_len = len(os.listdir(train19_imgroot))
    print("naic 2019 train {} images".format(tmp_19_len))
    train19_label_path = join(train19root, 'label.txt')
    with open(train19_label_path, 'r') as f19:
        with open(train_label_path, 'a') as f:
            labels19 = f19.readlines()
            pid_imgList_19 = defaultdict(list)
            for x in labels19:
                tmp_img_name, tmp_pid =  x.rstrip().split(":")
                pid_imgList_19[tmp_pid].append(tmp_img_name)
            for k, v in pid_imgList_19.items():
                for x in v:
                    shutil.copy(join(train19_imgroot, x), join(train_dest_root, x))
                    tmp_label = x + ':' + str(train_label) + '\n'
                    f.write(tmp_label)
                train_label += 1
    tmp_total_len = len(os.listdir(train_dest_root))
    print("our total train {} images".format(tmp_total_len))
    print('----'*8)
    print("copy 2020 gallery normal image to gallery path")
    counter = 1
    g20imgs = os.listdir(gallery_20_path)
    max_id = 0
    with open(test_label_path, 'r') as f:
        labels = f.readlines()
        max_id = len(labels)+1
    with open(test_label_path, 'a') as f:
        for x in g20imgs:
            if n_g_counter <= gallery_normal_num:
                g_or_n = jugde_normal_or_green(join(gallery_20_path, x))
                if g_or_n == 'normal':
                    shutil.copy(join(gallery_20_path, x), join(g_dest_root, x))
                    tmp_label = x + ':' + str(max_id + counter) + '\n'
                    f.write(tmp_label)
                    n_g_counter += 1
                    counter += 1
            else:
                break
    print("copy {} normal images to our gallery".format(counter))
    tmp_len = len(os.listdir(g_dest_root))
    print("our gallery has {} total images".format(tmp_len))
    print("finished!")


if __name__ == "__main__":
    label_txt = '/mnt/nfs-internstorage/train_data/naic2020/train/label.txt'
    img_source_root = '/mnt/nfs-internstorage/train_data/naic2020/train/images'
    dest_root = '/mnt/nfs-internstorage/train_data/naic2020_evaldataset/'
    main(label_txt, 
        img_source_root, 
        dest_root, 
         train19root='/mnt/nfs-internstorage/train_data/naic_2019train_2020val/train',
         test_20_path='/mnt/nfs-internstorage/train_data/naic2020/image_A/',
        query_normal_num=1524)
