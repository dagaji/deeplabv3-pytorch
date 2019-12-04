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
from deeplabv3.augmentation import Compose, ComposeAngle, Compose_v2
from .register import register
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import deeplabv3.lines as lines
import pickle
import deeplabv3.vis as vis
from scipy import ndimage
import pickle
from pathlib import Path

@register.attach('angle_detect_dataset')
class AngleDetectDataset(data.Dataset):

    def __init__(self, root, id_list_path, 
    	angle_step=15.0, min_angle=-30.0, max_angle=30.0, augmentations=[]):
    
        self.root = root
        self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.id_list = [img_id for img_id in self.id_list.tolist() if "APR" in img_id]
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = Compose(augmentations)
        self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        self.min_angle = min_angle
        self.max_angle = max_angle
        

    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx] + '.png'
        img_path = os.path.join(self.root, "images", image_id)
        label_path = os.path.join(self.root, "masks", image_id)
        label_test_path = os.path.join(self.root, "masks_test", image_id)

        img = np.asarray(imread(img_path))
        label = np.asarray(imread(label_path))[..., 0]
        label_test = np.asarray(imread(label_test_path))[..., 0]
        label_test[label_test == 255] = 0

        return image_id, img, label, label_test


    def __getitem__(self, index):

        image_id, img, label, label_test = self._load_data(index)

        image, _label = self.augmentations(img, np.dstack((label, label_test)))
        label, label_test = np.dsplit(_label, 2)
        label = np.squeeze(label)
        label_test = np.squeeze(label_test)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        # plt.figure()
        # plt.imshow(label_test == 1)

        _, angles_v, dists_v = lines.search_lines((label_test == 1), (self.min_angle, self.max_angle), npoints=1000, min_distance=100, min_angle=300, threshold=None)
        _rot_angle = -np.rad2deg(angles_v).mean()
        # print(">>>> angle: {}".format(_rot_angle))

        angle_range_gt = np.zeros(len(self.rot_angles) - 1, dtype=np.float32)
        angle_dist = np.abs(self.rot_angles - _rot_angle)
        angle_range_label = np.argsort(angle_dist)[:2].min()
            
        return dict(image_id=image_id, image=image, angle_range_label=angle_range_label)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str