import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch as t
from PIL import Image
from skimage import transform as sktsf
from sklearn.model_selection import train_test_split
from torchvision import transforms as tvtsf

from utils.config import opt, CLASS_NUMBER
from . import util
from .voc_dataset import VOCBboxDataset


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class CsvDB:
    def __init__(self, labels_len):
        self.label_names = []
        for i in range(labels_len):
            self.label_names.append('label' + str(i))


from torchvision.transforms import *

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

size = 224


class CsvDataset(object):
    def __init__(self, base, csv_path, transform=None):
        self._transform = transform
        self._base = base
        self._csv = pd.read_csv(csv_path)
        self._train, self._test = train_test_split(self._csv)
        self._mode = 'train'
        self.db = CsvDB(CLASS_NUMBER)
        if transform is None:
            self._transform = Compose([Resize((size, size)), ToTensor(), Normalize(rgb_mean, rgb_std)])

    def set_mode(self, mode):
        self._mode = mode

    def __len__(self):
        if self._mode == 'train':
            return len(self._train)
        return len(self._test)

    def __getitem__(self, item):
        if self._mode == 'train':
            target_csv = self._train
        else:
            target_csv = self._test

        img = Image.open(osp.join(self._base, target_csv.iloc[item, 0]))
        img_size = [size, size]
        img = self._transform(img)
        label = torch.FloatTensor([int(target_csv.iloc[item, 5])])
        y1 = float(target_csv.iloc[item, 1] * img_size[0])
        x1 = float(target_csv.iloc[item, 2] * img_size[1])
        y2 = float(target_csv.iloc[item, 3] * img_size[0])
        x2 = float(target_csv.iloc[item, 4] * img_size[1])
        box = torch.FloatTensor([x1, y1, x2, y2]).view(1, 4)
        if self._mode == 'train':
            return img, box, label, torch.FloatTensor([0.5])
        else:
            return img, torch.FloatTensor([img_size]).view((2, 1)), box, label, torch.FloatTensor([0])


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
