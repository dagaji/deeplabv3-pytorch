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
        # self.id_list = [img_id for img_id in self.id_list.tolist() if "APR" in img_id]
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
        label = np.squeeze(label).astype(np.int64)
        label_test = np.squeeze(label_test).astype(np.int64)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        angle_range_label = 255
        if np.any(label_test == 1):
            # _, angles_v, dists_v = lines.search_lines((label_test == 1), (self.min_angle, self.max_angle), npoints=1000, min_distance=100, min_angle=300, threshold=None)
            # _rot_angle = -np.rad2deg(angles_v).mean()
            _, _rot_angle = lines.extract_lines((label_test == 1), (self.min_angle, self.max_angle))
            _rot_angle = np.rad2deg(_rot_angle)
            angle_dist = np.abs(self.rot_angles - _rot_angle)
            angle_range_label = np.argsort(angle_dist)[:2].min()
            
        return dict(image_id=image_id, image=image, angle_range_label=angle_range_label, label_test=label_test, label=label)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


@register.attach('angle_detect_dataset_multitask')
class AngleDetectDatasetMultitask(data.Dataset):

    def __init__(self, **kwargs):
        self.theta_step = kwargs.pop('theta_step')
        self.rho_step = kwargs.pop('rho_step')
        super(AngleDetectDatasetMultitask, self).__init__(**kwargs)

    def get_line_gt(self, true_lines, proposed_lines, sz):

        n_lines = proposed_lines.shape[0]
        lines_gt = np.zeros(n_lines, dtype=np.float32)

        for idx in np.arange(n_lines):
            distance = np.abs(proposed_lines[idx] - true_lines)
            close_lines = np.logical_and(distance[:,0] < self.rho_step, distance[:,1] < np.deg2rad(5.0))
            if np.any(close_lines):
               lines_gt[idx] = 1.0

        return lines_gt


    def __getitem__(self, index):

        data = super(AngleDetectDatasetMultitask, self).__getitem__(index)
        label_test = data['label_test']
        label = data['label']
        idx = data['label_test']
        
        if idx != 255:

            angle_range_v = np.deg2rad((self.angles_v[idx], self.angles_v[idx+1]))
            angle_range_h = angle_range_v + np.pi/2
            sz = label_test.shape
            
            proposed_lines_v, line_endpoints_v = lines.get_line_proposals(angle_range_v, sz, angle_step=self.angle_step, rho_step=self.rho_step)
            true_lines_v, _ = lines.extract_lines((label_test == 1), (self.min_angle, self.max_angle))
            lines_gt_v = self.get_line_gt(true_lines_v, proposed_lines_v, sz)

            proposed_lines_h, line_endpoints_h = lines.get_line_proposals(angle_range_h, sz, angle_step=self.angle_step, rho_step=self.rho_step)
            true_lines_h, _ = lines.extract_lines((label_test == 1), (self.min_angle + 90.0, self.max_angle + 90.0))
            lines_gt_h = self.get_line_gt(true_lines_h, proposed_lines_h, sz)

            proposed_lines_endpoints = np.vstack((line_endpoints_v, line_endpoints_h), dtype=np.float32)
            lines_gt = np.append(lines_gt_v, lines_gt_h)

        label[(label != 255)] = np.clip(label[(label != 255)] - 1, a_min=0)

        data.update(proposed_lines_endpoints=proposed_lines_endpoints, lines_gt=lines_gt, label=label)

        return data

