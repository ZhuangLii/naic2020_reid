import os
from glob import glob
root = '/home/zjf/git/naic_person_reid_bak/data/train'
img_root = root + '/images'
imgs = glob(img_root+'/*.png')
imgs = [x.split('/')[8] for x in imgs]
from tqdm import tqdm
with open(root + '/label.txt') as f:
    labels = f.readlines()
for label in tqdm(labels):
    img_name = label.split(':')[0]
    if not img_name in imgs:
        print(img_name)


