import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Literal

import numpy
import torch
from PIL import Image
from kornia.geometry import vflip, hflip
from mmengine import ConfigDict
from torch import Tensor
from torch.utils import data
from torchvision import transforms
from torchvision.ops import box_convert
from torchvision.transforms import Resize

from loader.utils.checker import check_mask, check_iqa, get_max_size, check_image
from loader.utils.reader import gray_read, ycbcr_read, label_read
from models.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class MSRS_data(data.Dataset):
    color = True
    classes = ['person']
    palette = ['#FF0000']

    generate_meta_lock = False
    def __init__(self, root: str | Path, mode: Literal['train', 'val', 'pred'], config: ConfigDict, data_dir:str, transform=to_tensor):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.config = config

        dirname = os.listdir(data_dir)

        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path
            elif(sub_dir == "vi"):
                self.vis_path = temp_path

        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

        img_list = Path(f"{root}/meta/{mode}.txt").read_text().splitlines()
        self.img_list = img_list


        self.labels = self.check_labels(self.root, img_list)

        match mode:
            case _:
                self.max_size = get_max_size(Path(root), img_list)
                self.transform_fn = Resize(size=self.max_size)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index: int) -> dict:
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        match self.mode:
            case 'train' | 'val':
                sample = self.train_val_item(index)
                sample["vi"] = vis_image
                return sample
            case "pred":
                sample = self.pred_item(index)
                sample["vi"] = vis_image
                print("GGGGGG", sample.keys())
                return sample



    def check_labels(self, root: Path, img_list: List[str]) -> List[Tensor]:

        labels = []
        for img_name in img_list:
            label_name = Path(img_name).stem + '.txt'
            labels.append(self.label_read(f"{root}/labels/{label_name}"))
        logging.info('find all labels on list')
        return labels

    def label_read(self, label_path: str | Path) -> Tensor:
        target = numpy.loadtxt(str(label_path), dtype=numpy.float32)
        labels = torch.from_numpy(target).view(-1, 5)
        labels[:, 1:] = box_convert(labels[:, 1:], 'cxcywh', 'xyxy')
        return labels

    def train_val_item(self, index: int) -> dict:
        name = self.img_list[index]
        logging.debug(f'train-val mode: loading item {name}')
        ir = gray_read(f"{self.root}/ir/{name}")
        vi_y, cb, cr = ycbcr_read(f"{self.root}/vi/{name}")
        label_p = Path(name).stem + '.txt'
        labels = label_read(f"{self.root}/labels/{label_p}")
        t = torch.cat([ir, vi_y, cb, cr], dim=0)
        resize_fn = Resize(size=self.config.train.image_size)
        t = resize_fn(t)
        labels_o = torch.zeros((len(labels), 6))
        if len(labels):
            labels_o[:, 1:] = labels
        ir, vi_y, cb, cr = torch.split(t, [1, 1, 1, 1], dim=0)
        sample = {
            'name': name,
            'ir': ir,
            'vi_y': vi_y,
            'cb': cb,
            "cr":cr,
            'labels': labels_o
        }
        return sample

    def pred_item(self, index: int) -> dict:
        name = self.img_list[index]
        logging.debug(f'pred mode: loading item {name}')
        ir = gray_read(f"{self.root}/ir/{name}")
        vi, cb, cr = ycbcr_read(f"{self.root}/vi/{name}")
        s = ir.shape[1:]
        t = torch.cat([ir, vi, cb, cr], dim=0)
        ir, vi, cbcr = torch.split(self.transform_fn(t), [1, 1, 2], dim=0)
        sample = {'name': name, 'ir': ir, 'vi_y': vi, 'cbcr': cbcr, 'shape': s}
        return sample

    @staticmethod
    def collate_fn(data: List[dict]) -> dict:
        keys = data[0].keys()
        new_data = {}
        for key in keys:
            k_data = [d[key] for d in data]
            match key:
                case 'name' | 'shape':
                    new_data[key] = k_data
                case 'labels':
                    for i, lb in enumerate(k_data):
                        lb[:, 0] = i
                    new_data[key] = torch.cat(k_data, dim=0)
                case _:
                    new_data[key] = torch.stack(k_data, dim=0)
        return new_data

