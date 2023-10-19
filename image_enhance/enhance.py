import os
import cv2
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from image_enhance.model import Finetunemodel
import logging
from tqdm import tqdm
from image_enhance.multi_read_data import MemoryFriendlyLoader
from pathlib import Path
from time import sleep
from tqdm import tqdm

import matplotlib as mpl

mpl.rcParams['font.family'] = 'STKAITI'
plt.rcParams['axes.unicode_minus'] = False
def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def load_model(model_list, brightness):
    if brightness < 30:
        return model_list[0]
    elif 30 <= brightness <= 40:
        return model_list[1]
    elif brightness > 70:
        return model_list[3]
    else:
        return model_list[2]

def enhance_main(test_queue, model_list, save_path):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    for i, (input, image_name) in enumerate(test_queue):
        image = torch.squeeze(input).numpy().transpose(1, 2, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0] * 255
        print(brightness)

        model = load_model(model_list, brightness)

        model_1 = Finetunemodel(model)
        model_2 = model_1.cuda()

        model_2.eval()

        with torch.no_grad():
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('\\')[-1].split('.')[0]

            i, r = model_2(input)
            u_name = '%s.png' % (image_name)
            u_path = save_path + '/' + u_name
            save_images(r, u_path)
    print("Image enhance finish")


if __name__ == '__main__':
    model_list = [
        "./weights/medium.pt",
        "./weights/difficult.pt",
        "./weights/easy.pt"
    ]

