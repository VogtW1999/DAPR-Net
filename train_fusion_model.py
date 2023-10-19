# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from kornia.color import ycbcr_to_rgb

from config import from_dict
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from NewMove.detect import Detect
from data_loader.msrs_data import MSRS_data
from models.common import gradient, clamp, YCrCb2RGB
from models.fusion_model import DAPR
from image_enhance.enhance import enhance_main
from image_enhance.multi_read_data import MemoryFriendlyLoader
from image_enhance.model import Finetunemodel
from pathlib import Path

import matplotlib as mpl



def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DAPR')
    parser.add_argument('--dataset_path', metavar='DIR', default='./datasets/llvip',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model', choices=['fusion_model'])
    parser.add_argument('--save_model_path', default='pretrained')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int, metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[3, 7, 50, 50]', type=str, metavar='N', help='loss weight')
    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth', help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool, help='use GPU or not.')
    parser.add_argument('--EnhanceImgPath', default="./datasets/llvip/Vis", help='after')
    parser.add_argument('--OriginImgPath', default="./datasets/llvip/Vis_", help='before')
    parser.add_argument('--ModelList',
                        default="./image_enhance/weights/medium.pt,./image_enhance/weights/difficult.pt,./image_enhance/weights/easy.pt,./image_enhance/weights/none.pt",help='图像增强前图片路径')
    parser.add_argument('--data_path', type=str, default=r'./datasets/llvip/Vis_', help='location of the data corpus')
    parser.add_argument('--save_path', type=str, default=r'./datasets/llvip/Vis', help='location of the data corpus')
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')

    args = parser.parse_args()

    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)
    config = config

    init_seeds(args.seed)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    test_queue = torch .utils.data.DataLoader(
        TestDataset, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=0, generator=torch.Generator())
    train_dataset = MSRS_data(root=config.dataset.root, mode='train', config=config, data_dir=args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, pin_memory=True, num_workers=0,
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    enhance_img_list = Path(args.EnhanceImgPath).rglob("*.png")
    origin_img_list = Path(args.OriginImgPath).rglob("*.png")
    if not (len(list(enhance_img_list)) == len(list(origin_img_list))):
        logging.info("Image enhacne begin...")
        enhance_main(test_queue, (args.ModelList).split(","), save_path)

    if args.arch == 'fusion_model':
        model = DAPR()
        model = model.cuda()
        detect_model = Detect(config, mode='train', nc=len(train_dataset.classes), classes=train_dataset.classes, labels=train_dataset.labels)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss = 0
        loss_list = []
        ckpt_list = []
        temp_loss = 1e6
        loss_total_picture = []
        final_save_model_path = args.save_model_path + "/" + time.strftime("%Y-%m-%d-%H-%M-%S")
        if final_save_model_path:
            os.makedirs(final_save_model_path)

        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            for sample in train_tqdm:
                vis_y_image = sample["vi_y"].cuda()
                vis_image = sample["vi"].cuda()
                inf_image = sample["ir"].cuda()
                cb_image = sample["cb"].cuda()
                cr_image = sample["cr"].cuda()

                optimizer.zero_grad()
                fused_image = model(vis_y_image, inf_image)
                fused_image_1 = fused_image = clamp(fused_image)

                if train_dataset.color:
                    fused_image_2 = YCrCb2RGB(fused_image[0], cb_image[0], cr_image[0])
                    unloader = transforms.ToPILImage()(fused_image_2)
                    unloader.save(f'fuse_pictures/{sample["name"][0]}')
                    fused_image = torch.cat([fused_image, sample['cb'].cuda(), sample['cr'].cuda()], dim=1)
                    fused_image = ycbcr_to_rgb(fused_image)
                d_loss, [box_l, obj_l, cls_l] = detect_model.criterion(
                    imgs=fused_image,
                    targets=sample['labels'],
                )
                gradinet_loss = F.l1_loss(gradient(fused_image_1), torch.max(gradient(inf_image), gradient(vis_y_image)))
                t1, t2, t3 ,t4 = eval(args.loss_weight)
                loss =  t2 * loss_aux + t3 * gradinet_loss + t4 * d_loss

                train_tqdm.set_postfix(epoch=epoch, loss_illum=None, loss_aux=t2 * loss_aux.item(),
                                       gradinet_loss=t3 * gradinet_loss.item(),d_loss =t4 * d_loss.item(),
                                       loss_total=loss.item())

                seen_x, preview = detect_model.eval(sample=sample["name"], imgs=fused_image, targets=sample['labels'].cuda(),
                                                   stats=[], preview=True)


                loss.backward()
                optimizer.step()


            loss_total_picture.append(loss.item())
            if loss.item() <= temp_loss:
                temp_loss = loss.item()

                loss_list.append(temp_loss)
                ckpt_list.append(model.state_dict())
                if len(loss_list) > 5 or len(ckpt_list) > 5:
                    file_path = f'{final_save_model_path}/fusion_model_epoch_{epoch}.pth'
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    ckpt_list.pop(0)
                    loss_list.pop(0)
                loss_list.append([temp_loss, epoch])
                logging.info("begining save model")
                for ckpt in ckpt_list:
                    torch.save(model.state_dict(), f'{final_save_model_path}/fusion_model_epoch_{epoch}.pth')
                logging.info(f'Epoch {epoch}/{args.epochs} | Model Saved')

