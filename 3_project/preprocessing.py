import os
import re

import cv2
import numpy as np
import pandas as pd


## HELPER FUNCTIONS
import torch
from torchvision.io import write_video

def load_images_from_folder(folder, filter=None, masking=False):
    images = []
    mask = cv2.imread('data/mask.png', 0)
    for filename in os.listdir(folder):
        img = None
        if filter is None:
            # print('?')
            img = cv2.imread(os.path.join(folder, filename))
            # apply masking
            if masking:
                img = cv2.bitwise_and(img, img, mask=mask)
        else:
            if filename in filter:
                # print(filename)
                img = cv2.imread(os.path.join(folder, filename))
                if masking:
                    img = cv2.bitwise_and(img, img, mask=mask)

        if img is not None:
            images.append(img)
    # print('Loaded {} images'.format(len(images)))
    return images


def makenumpy(df):
    m = None
    for r in df:
        if m is None:
            m = np.array(r).reshape(1, -1)
        else:
            m = np.concatenate((m, np.array(r).reshape(1, -1)), axis=0)
    return m


def merg_v1(warped):
    c = (warped > 0).sum(axis=0)
    warped_ = warped.sum(axis=0).squeeze()
    c[c == 0] = 1.0
    c = 1.0 / c
    return c * warped_


def merg_v2(warped):
    arr = np.zeros((warped[0].shape[0], warped[0].shape[1], 3))
    divide_arr = np.ones((warped[0].shape[0], warped[0].shape[1]))

    for l in range(len(warped)):
        (x, y, c) = (np.where(warped[l] != 0.0))
        divide_arr[x, y] += 1
        arr += warped[l]

    (x, y, c) = (np.where(arr == 0.0))
    divide_arr[x, y] = 1
    arr[:, :, 0] /= divide_arr
    arr[:, :, 1] /= divide_arr
    arr[:, :, 2] /= divide_arr

    return arr


def merg_avg(warped):
    return np.array(warped).mean(axis=0)


#######################################
# Folder Creation
cwd = os.getcwd()
out_dir = os.path.join(cwd, "data_preprocessed")

dirs = ["train", "validation", "test"]
fns = ["train", "valid", "test"]
methods = ["merged", "bri_cont", "invert", "invert_video"]

for f in methods:
    for sub_f in fns:
        os.makedirs(os.path.join(out_dir, f, sub_f), exist_ok=True)

## HELPER FOR IMAGE TRANSFORMATION
from torchvision.transforms import functional, ToTensor, Lambda, Compose
from torchvision.utils import save_image

tensorizer = ToTensor()

brightness_factor = 0.3
contrast_factor = 4
bri_cont_transform = Compose([
    Lambda(lambda x: functional.adjust_brightness(x, brightness_factor)),
    Lambda(lambda x: functional.adjust_contrast(x, contrast_factor))])

# bri_cont_transform = ColorJitter(brightness=0.4, contrast=3)
inverter_transform = Lambda(lambda x: functional.invert(x))

# IMAGE PREPROCESSING
time_frames = 7
mask = cv2.imread('data/mask.png', 0)

for i, sdir in enumerate(dirs):
    train_folder_pattern = re.compile(fns[i] + '-(.*)-(.*)')
    for folder in os.listdir(os.path.join('data', sdir)):
        if train_folder_pattern.match(folder) is not None:
            print('prc Folder:', folder, end='')

            homo = pd.read_json('data/' + sdir + '/' + folder + '/homographies.json', orient='record')

            for tf in range(time_frames):
                warped = []
                for fn in os.listdir('data/' + sdir + '/' + folder):
                    if ('merg' not in fn) and ('.png' in fn) and ((str(tf) + '-') in fn):
                        print(' ', fn, end='')
                        img_f = './data/' + sdir + '/' + folder + '/' + fn
                        img = cv2.imread(img_f)

                        img = cv2.bitwise_and(img, img, mask=mask)
                        h = makenumpy(homo[fn.replace('.png', '')])

                        warped += [cv2.warpPerspective(img / 255., h, img.shape[:2])]
                d = {"merg_avg_masked": merg_avg(warped)}  # ,
                # "merg_v1_masked": merg_v1(np.array(warped)),
                # "merg_v2_masked": merg_v2(warped)}

                for k, v in d.items():
                    v = tensorizer(v)
                    save_image(v,
                               os.path.join(out_dir, "merged", fns[i], f'{folder.split("-", 1)[-1]}_{str(tf)}-{k}.png'))

                    bri_cont = bri_cont_transform(v)
                    save_image(bri_cont,
                               os.path.join(out_dir, "bri_cont", fns[i],
                                            f'{folder.split("-", 1)[-1]}_{str(tf)}-{k}-bri-cont.png'))

                    inverted = inverter_transform(bri_cont)
                    save_image(inverted,
                               os.path.join(out_dir, "invert", fns[i],
                                            f'{folder.split("-", 1)[-1]}_{str(tf)}-{k}-invert.png'))

            print('\n')
