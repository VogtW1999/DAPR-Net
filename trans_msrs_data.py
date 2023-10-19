# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DAPR')
    parser.add_argument('--h5_path', metavar='DIR', default='datasets/data_MSRS.h5',
                        help='path to dataset')
    parser.add_argument('--msrs_root_path', metavar='DIR', default='datasets/msrs_train',
                        help='path to dataset')
    args = parser.parse_args()

    h5_path = args.h5_path
    msrs_root_path = args.msrs_root_path

    f = h5py.File(h5_path, 'r')
    sources = f['data'][:]
    sources = np.transpose(sources, (0, 3, 2, 1))
    vi_images = sources[:, :, :, 0:3]
    ir_images = sources[:, :, :, 3:4]
    day_iter = 0
    night_iter = 0

    vis_dir = os.path.join(msrs_root_path, 'Vis')
    Inf_dir = os.path.join(msrs_root_path, 'Inf')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(Inf_dir):
        os.makedirs(Inf_dir)

    for index, (vi_image, ir_image) in enumerate(tqdm(zip(vi_images, ir_images), total=vi_images.shape[0])):
        vi_image = np.uint8(vi_image * 255)
        ir_image = np.uint8(ir_image * 255)
        vi_image = cv2.cvtColor(vi_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(vis_dir, f'{index}.png'), vi_image)
        cv2.imwrite(os.path.join(Inf_dir, f'{index}.png'), ir_image)
