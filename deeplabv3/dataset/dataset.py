import matplotlib
matplotlib.use('tkagg')
import random
import numpy as np
import torch
from torch.utils import data
import os.path
import os
import pdb
from torchvision import transforms
from skimage.io import imread
import torchvision.transforms.functional as TF
from deeplabv3.augmentation import Compose, ComposeAngle
from .register import register
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import deeplabv3.lines as lines
import pickle
import deeplabv3.vis as vis



@register.attach('base_dataset')
class BaseDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(self, root, id_list_path, augmentations=[], masks_test=True, change_ignore_index=False):
        self.root = root
        self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = Compose(augmentations)
        self.masks_test = masks_test
        self.change_ignore_index = change_ignore_index


    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx] + '.png'
        img_path = os.path.join(self.root, "images", image_id)
        label_path = os.path.join(self.root, "masks_test" if self.masks_test else "masks", image_id)

        img = np.asarray(imread(img_path))
        label = np.asarray(imread(label_path))[..., 0]
        return image_id, img, label


    def __getitem__(self, index):

        image_id, image, label = self._load_data(index)
        image, label = self.augmentations(image, label)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)

        label = label.astype(np.int64)
        image = image.numpy()
        if self.change_ignore_index:
            label[label == 255] = -100

        return dict(image_id=image_id, image=image, label=label)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


@register.attach('slim_dataset')
class SlimDataset(BaseDataset):

    kernel = np.ones((5,5), np.uint8)
    iterations = 5

    def __init__(self, root, id_list_path, augmentations=[], change_ignore_index=False, train=True):
        super(SlimDataset, self).__init__(root, 
                                          id_list_path, 
                                          augmentations=augmentations, 
                                          masks_test=True, 
                                          change_ignore_index=change_ignore_index)
        self.train = train

    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id, img, label = super(SlimDataset, self)._load_data(idx)
        # plt.figure()
        # plt.imshow(label)
        label_1 = (label == 1).astype(np.uint8)
        label_255 = np.logical_or(label == 1, label == 255).astype(np.uint8)
        label[label_255 > 0] = 0
        if not self.train:
            label_255 = cv2.dilate(label_255, self.kernel, iterations=self.iterations)
            label[label_255 > 0] = 255
        label[label_1 > 0] = 1
        # plt.figure()
        # plt.imshow(label)
        # plt.show()
        return image_id, img, label


@register.attach('multitask')
class MultiTaskDataset(BaseDataset):

    ignore_label = -100

    def __init__(self, root, id_list_path, augmentations=[], masks_test=False, change_ignore_index=False):
        super(MultiTaskDataset, self).__init__(root, id_list_path, augmentations, masks_test=False, change_ignore_index=False)

    def __getitem__(self, index):
        data = super(MultiTaskDataset, self).__getitem__(index)
        label = data['label']

        mask = (label != 255)
        label_3c = label.copy()
        label_3c[label == 0] = 1
        label_3c[mask] -= 1

        label_2c = label.copy()
        label_2c[np.logical_or(label == 2, label == 3)] = 255

        label_3c[label_3c == 255] = self.ignore_label
        label_2c[label_2c == 255] = self.ignore_label

        data.update(label_3c=label_3c, label_2c=label_2c)
        return data


@register.attach('hist')
class HistDataset(data.Dataset):

    """
    Base dataset class
    """
    min_angle = -40
    max_angle = 40
    angle_step = 10
    angles1 = np.arange(min_angle, max_angle + angle_step, angle_step)
    angles2 = angles1 + 90
    ignore_label = -100

    def __init__(self, root, id_list_path, augmentations=[], masks_test=False, APR_only=True):
        self.root = root
        with open(os.path.join(root, 'angles.pickle'), "rb") as file:
            self.angles_dict = pickle.load(file)
        #self.id_list = np.loadtxt(id_list_path, dtype=str)
        if APR_only:
            id_list = np.loadtxt(id_list_path, dtype=str)
            self.id_list = []
            for _id in id_list:
                if "APR" in _id:
                    self.id_list.append(_id)
        else:
            self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = ComposeAngle(augmentations)
        self.masks_test = masks_test


    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx] + '.png'
        img_path = os.path.join(self.root, "images", image_id)
        label_path = os.path.join(self.root, "masks_test" if self.masks_test else "masks", image_id)

        img = np.asarray(imread(img_path))
        label = np.asarray(imread(label_path))
        if len(label.shape) == 3:
            label = label[...,-1]
        angles = self.angles_dict.get(image_id, None)

        return image_id, img, label, angles


    def __getitem__(self, index):

        image_id, image, label, angles = self._load_data(index)
        image, label, angles = self.augmentations(image, label, angles)
        hist_gt = np.argmin(np.abs(self.angles1 - angles[1])).astype(np.int64)

        label = label.astype(np.int64)

        mask = (label != 255)
        label_3c = label.copy()
        label_3c[label == 0] = 1
        label_3c[mask] -= 1

        label_2c = label.copy()
        label_2c[np.logical_or(label == 2, label == 3)] = 255

        label_3c[label_3c == 255] = self.ignore_label
        label_2c[label_2c == 255] = self.ignore_label

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        return dict(image_id=image_id, 
                    image=image, 
                    label=label, 
                    hist_gt=hist_gt,
                    label_3c=label_3c,
                    label_2c=label_2c)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


@register.attach('histv2')
class HistDatasetv2(data.Dataset):

    """
    Base dataset class
    """
    min_angle = -40
    max_angle = 40
    angle_step = 10
    angles1 = np.arange(min_angle, max_angle + angle_step, angle_step)
    angles2 = angles1 + 90
    ignore_label = -100

    def __init__(self, root, id_list_path, augmentations=[], masks_test=False, APR_only=False):
        self.root = root
        with open(os.path.join(root, 'angles.pickle'), "rb") as file:
            self.angles_dict = pickle.load(file)
        #self.id_list = np.loadtxt(id_list_path, dtype=str)
        if APR_only:
            id_list = np.loadtxt(id_list_path, dtype=str)
            self.id_list = []
            for _id in id_list:
                if "APR" in _id:
                    self.id_list.append(_id)
        else:
            self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = ComposeAngle(augmentations)
        self.masks_test = masks_test


    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx] + '.png'
        img_path = os.path.join(self.root, "images", image_id)
        label_path = os.path.join(self.root, "masks_test" if self.masks_test else "masks", image_id)

        img = np.asarray(imread(img_path))
        label = np.asarray(imread(label_path))
        if len(label.shape) == 3:
            label = label[...,-1]
        angles = self.angles_dict.get(image_id, None)

        return image_id, img, label, angles


    def __getitem__(self, index):

        image_id, image, label, angles = self._load_data(index)
        image, label, angles = self.augmentations(image, label, angles)
        hist_gt = np.zeros(len(self.angles1), dtype=np.float32)
        if angles is not None:
            idx = np.argmin(np.abs(self.angles1 - angles[1]))
            hist_gt[idx] = 1.0
            error = angles[1] - self.angles1[idx]
            if abs(error) > 4.0:
                other_idx = idx + (error > 0) - (error < 0)
                hist_gt[other_idx] = 1.0

        label = label.astype(np.int64)

        mask = (label != 255)
        label_3c = label.copy()
        label_3c[label == 0] = 1
        label_3c[mask] -= 1

        label_2c = label.copy()
        label_2c[np.logical_or(label == 2, label == 3)] = 255

        label_3c[label_3c == 255] = self.ignore_label
        label_2c[label_2c == 255] = self.ignore_label

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        return dict(image_id=image_id, 
                    image=image, 
                    label=label, 
                    hist_gt=hist_gt,
                    label_3c=label_3c,
                    label_2c=label_2c)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str




