import os
from tqdm import tqdm
train_img_root = '/mnt/nfs-internstorage/user/zjf/trainset2020/train/images'
train_label_path = '/mnt/nfs-internstorage/user/zjf/trainset2020/train/label.txt'
with open(train_label_path, 'r') as f:
    labels = f.readlines()
imgs = os.listdir(train_img_root)

print("-----"*8)
if len(set(imgs)) == len(imgs):
    print("no overlap in images")
else:
    print("some images have same names!!!")
print("-----"*8)
img_name_from_label = [x.rstrip().split(":")[0] for x in labels]
if len(set(img_name_from_label)) == len(img_name_from_label):
    print("no overlap image name in label")
else:
    print("some images have different pid in labels !!!")
print("-----"*8)
print("total images {}, total {} label images".format(len(imgs), len(img_name_from_label)))
print("total label images - total images {}".format(len(imgs)-len(img_name_from_label)))
counter = 0
left = []
for x in tqdm(imgs):
    if x not in img_name_from_label:
        counter += 1
        left.append(x)
# print("{} images in images not in label".format(len(set(img_name_from_label).difference(set(imgs)))))
print("{} images in images not in label".format(counter))
print("show some example", left[:10])
