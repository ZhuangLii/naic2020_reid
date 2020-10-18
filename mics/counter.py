import os
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def compute(path):
    file_names = os.listdir(path)
    gcounter = ncounter = 0
    for file_name in tqdm(file_names):
        img = cv2.imread(os.path.join(path, file_name))
        per_image_Bmean = np.mean(img[:, :, 0])
        per_image_Gmean = np.mean(img[:, :, 1])
        per_image_Rmean = np.mean(img[:, :, 2])
        if per_image_Bmean > 65 and per_image_Gmean > 65 and per_image_Rmean > 65:
            # file_name_green.append(file_name)
            gcounter += 1
        else:
            # file_name_none.append(file_name)
            ncounter += 1
    print('gcounter = ', gcounter)
    print('ncounter = ', ncounter)



if __name__ == '__main__':
    parser = ArgumentParser(description='vis txt result Tool')
    parser.add_argument('--path', '-p', help='dir to the query datasets')
    args = parser.parse_args()
    compute(args.path)


