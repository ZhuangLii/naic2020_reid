
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

filepath = '/home/zjf/naic_code/data/train/images'  # the path of dataset
pathDir = os.listdir(filepath)

R_channel = 0
G_channel = 0
B_channel = 0
img_size = 0

from tqdm import tqdm

for idx in tqdm(range(len(pathDir))):
    filename = pathDir[idx]
    img = cv2.imread(os.path.join(filepath, filename), cv2.COLOR_BGR2RGB)
    img = img/255.0
    img_size = img_size + img.shape[0]*img.shape[1]
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

R_mean = R_channel / img_size
G_mean = G_channel / img_size
B_mean = B_channel / img_size

R_channel = 0
G_channel = 0
B_channel = 0
for idx in tqdm(range(len(pathDir))):
    filename = pathDir[idx]
    img = cv2.imread(os.path.join(filepath, filename), cv2.COLOR_BGR2RGB)
    img = img/255.0
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean)**2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean)**2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean)**2)

R_var = (R_channel / img_size)**0.5
G_var = (G_channel / img_size)**0.5
B_var = (B_channel / img_size)**0.5

print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
