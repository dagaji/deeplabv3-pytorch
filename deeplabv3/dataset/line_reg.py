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

@register.attach('hist_dataset')
class HistDataset(data.Dataset):

    def __init__(self, root, id_list_path, angle_step=15.0, min_angle=-30.0, max_angle=30.0, augmentations=[], angle_step_sampler=5.0, rho_step_sampler=100, nlines=7):
        self.root = root
        self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = Compose(augmentations)
        self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.line_sampler = lines.LineSampler(angle_step=angle_step_sampler, rho_step=rho_step_sampler)
        self.rho_step_sampler = rho_step_sampler
        self.nlines = nlines
        self.sample_resol = rho_step_sampler / (nlines + 2)
        self.setup()

    def setup(self,):
        self.id_list = [img_id for img_id in self.id_list.tolist() if "APR" in img_id]


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

    def get_line_gt(self, true_lines, proposed_lines, sz):

        def _compute_iou(true_mask, proposed_mask):
            not_ignored = np.logical_and(true_mask > 0, true_mask < 2)
            hist = np.bincount(2 * true_mask[not_ignored].flatten() + proposed_mask[not_ignored].flatten(), minlength=4).reshape((2,2))
            iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            return iou[1]


        def _get_entropy(true_lines, true_masks, proposed_line):

            opt_lines = []
            ious = []
            proposed_rho, proposed_theta = proposed_line

            entropy_list = []
            opt_iou_list = []
            for idx in np.arange(true_masks.shape[0]):

                _true_line_coeffs = lines.general_form(*true_lines[idx].tolist())
                intersect_borders = np.array(lines.find_intesect_borders(_true_line_coeffs, sz))
                mid_point = intersect_borders.mean(0)
                opt_rho = mid_point[0] * np.cos(proposed_theta) + mid_point[1] * np.sin(proposed_theta)
                opt_line = (opt_rho, proposed_theta)
                opt_mask = _create_mask(opt_line, guard=False, width=16)
                opt_iou = _compute_iou(true_masks[idx], opt_mask)

                entropy = np.zeros(self.nlines, dtype=np.float32)
                for i in np.arange(self.nlines):
                    _rho = proposed_rho - self.sample_resol * (self.nlines - 1) / 2 + i * self.sample_resol
                    proposed_mask = _create_mask((_rho, proposed_theta), guard=False, width=16)
                    entropy[i] = _compute_iou(true_masks[idx], proposed_mask)
                entropy /= (entropy.sum() + 1e-5)

                opt_iou_list.append(opt_iou)
                entropy_list.append(entropy)

            entropy = entropy_list[np.argmax(opt_iou_list)]
            if np.any(np.isnan(entropy)):
                pdb.set_trace()
            if entropy.sum() < 0.5:
                return None
            return entropy

        def _plot_line(true_mask, proposed_mask, opt_mask, iou):
            vis_mask = opt_mask.copy()
            vis_mask[true_mask == 1] = 2
            vis_mask[proposed_mask == 1] = 3
            plt.figure()
            plt.imshow(vis_mask)
            plt.title('IoU: {}'.format(iou))

        def _create_mask(_line, width=8 , guard=True):

            line_mask = lines.create_grid(sz, [_line], width=width)
            if guard:
                return 2  * line_mask - lines.create_grid(sz, [_line], width=2*width)
            return line_mask

        true_lines_masks = np.array([_create_mask(_line, guard=True) for _line in true_lines], dtype=int)
        true_lines = np.array(true_lines, dtype=np.float32)
        proposed_lines = np.array(proposed_lines, dtype=np.float32)

        n_lines = proposed_lines.shape[0]
        lines_gt = np.zeros(n_lines, dtype=np.float32)
        entropy_gt = np.zeros((n_lines, self.nlines), dtype=np.float32)

        for idx in np.arange(n_lines):
            distance = np.abs(proposed_lines[idx] - true_lines)
            close_lines = np.logical_and(distance[:,0] < (self.rho_step_sampler / 2), distance[:,1] < np.deg2rad(5.0))
            if np.any(close_lines):
                entropy = _get_entropy(true_lines[close_lines], true_lines_masks[close_lines], proposed_lines[idx])
                if entropy is not None:
                    lines_gt[idx] = 1.0
                    entropy_gt[idx] = entropy

        return lines_gt, entropy_gt


    def __getitem__(self, index):

        image_id, img, label, label_test = self._load_data(index)

        image, _label = self.augmentations(img, np.dstack((label, label_test)))
        label, label_test = np.dsplit(_label, 2)
        label = np.squeeze(label)
        label_test = np.squeeze(label_test)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        seg_label = label.astype(np.int64)
        # line_points = np.zeros((200, 200, 2), dtype=np.float32)

        label_test = (label_test == 1)
        if np.any(label_test):

            _, angles_v, dists_v = lines.search_lines(label_test, (self.min_angle, self.max_angle), npoints=1000, min_distance=100, min_angle=300, threshold=None)
            true_lines_v = lines.get_lines(dists_v, angles_v)
            _, angles_h, dists_h = lines.search_lines(label_test, (self.min_angle + 90, self.max_angle + 90), npoints=1000, min_distance=100, min_angle=300, threshold=None)
            true_lines_h = lines.get_lines(dists_h, angles_h)
            
            angle_range_gt = np.zeros(len(self.rot_angles) - 1, dtype=np.float32)
            _rot_angle = -np.rad2deg(angles_v).mean()
            angle_dist = np.abs(self.rot_angles - _rot_angle)
            angle_range_gt[np.argsort(angle_dist)[:2].min()] = 1.0
            angle_range_v = self.rot_angles[np.argsort(angle_dist)[:2]]
            angle_range_gt[np.argsort(angle_dist)[:2]] = 1.0
            angle_range_v.sort()
            angle_range_h = angle_range_v + 90.0
            pdb.set_trace()

            # from deeplabv3.model.angle_detect import gabor
            # print(_rot_angle)
            # plt.figure()
            # plt.imshow(label_test)
            # plt.figure()
            # plt.imshow(gabor(np.pi/2 - np.deg2rad(_rot_angle)))
            # plt.figure()
            # plt.imshow(gabor(np.pi/2 - np.deg2rad(_rot_angle + 90)))
            # plt.show()

            proposed_lines_v, intersects_points_v = self.line_sampler(angle_range_v, label_test.shape)
            proposed_lines_h, intersects_points_h = self.line_sampler(angle_range_h, label_test.shape)

            lines_gt_v, entropy_gt_v = self.get_line_gt(true_lines_v, proposed_lines_v, label_test.shape)
            lines_gt_h, entropy_gt_h = self.get_line_gt(true_lines_h, proposed_lines_h, label_test.shape)

            lines_gt = np.append(lines_gt_v, lines_gt_h)
            entropy_gt = np.vstack((entropy_gt_v, entropy_gt_h))
            proposed_lines = np.array(proposed_lines_v + proposed_lines_h, dtype=np.float32)
            intersects_points = np.array(intersects_points_v + intersects_points_h, dtype=np.float32)
            
        return dict(image_id=image_id, image=image, seg_label=seg_label, intersects_points=intersects_points, lines_gt=lines_gt, entropy_gt=entropy_gt, angle_range_gt=angle_range_gt)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str