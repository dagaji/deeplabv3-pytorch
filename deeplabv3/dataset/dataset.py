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
from deeplabv3.augmentation import Compose
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


@register.attach('one_hot')
class OneHotDataset(BaseDataset):

    one_hot_matrix = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

    def __init__(self, root, id_list_path, augmentations=[], masks_test=False, change_ignore_index=False):
        super(OneHotDataset, self).__init__(root, id_list_path, augmentations, masks_test=False, change_ignore_index=False)

    def one_hot_encode(self, label):
        """Return one hot encoded labels."""
        _label = label.copy()
        _label[label == 255] = 4
        label_1hot = self.one_hot_matrix[_label]
        return label_1hot

    def __getitem__(self, index):
        data = super(OneHotDataset, self).__getitem__(index)
        label = data['label']
        label_1hot = self.one_hot_encode(label).astype(np.float32)
        label_1hot = np.transpose(label_1hot, (2,0,1))
        mask = (label != 255).astype(np.float32)
        data.update(label_1hot=label_1hot)
        data.update(mask=mask)
        return data


@register.attach('refine')
class RefineDataset(OneHotDataset):

    kernel = np.ones((3,3), np.uint8)
    iterations = 3

    def create_refine_mask(self, label):
        label_1 = (label == 1).astype(np.uint8)
        label_1 = cv2.erode(label_1, self.kernel, iterations=self.iterations) 
        refine_mask = -100 * np.ones(label.shape, dtype=np.int64)
        refine_mask[label_1 == 1] = 1
        return refine_mask

    def __getitem__(self, index):
        data = super(RefineDataset, self).__getitem__(index)
        label = data['label']
        refine_mask = self.create_refine_mask(label)
        data.update(refine_mask=refine_mask)
        return data

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

@register.attach('multitask+weights')
class MultiTaskWeightsDataset(MultiTaskDataset):

    kernel = np.ones((5,5), np.uint8)
    iterations = 6

    def __init__(self, root, id_list_path, augmentations=[], masks_test=False, change_ignore_index=False, T=0.3):
        super(MultiTaskWeightsDataset, self).__init__(root, id_list_path, augmentations, masks_test=masks_test, change_ignore_index=change_ignore_index)
        self.T = T

    def __getitem__(self, index):
        data = super(MultiTaskWeightsDataset, self).__getitem__(index)

        label_2c = data['label_2c']
        weights = np.zeros(label_2c.shape, dtype=np.float32)
        if np.any(label_2c == 1):
            label_2c_aux = cv2.dilate((label_2c == 1).astype(np.uint8), self.kernel, iterations=self.iterations)
            dt = ndimage.distance_transform_edt(label_2c_aux == 0)
            dt /= np.max(dt)
            dt = np.clip(dt, a_min=0, a_max=self.T)
            weights = np.exp(-dt)
            weights[label_2c == self.ignore_label] = 0
            weights = weights.astype(np.float32)
        label_2c[label_2c == self.ignore_label] = 0

        # plt.figure()
        # plt.imshow(weights)
        # plt.figure()
        # plt.imshow(label_2c)
        # plt.show()

        data.update(weights=weights, label_2c=label_2c)
        return data

@register.attach('instance')
class InstanceDataset(BaseDataset):

    instance_dataset_path = '/home/davidgj/projects/instance_APR'
    vis_dir = '/home/davidgj/projects/vis_aux'

    def _load_data(self, idx):
        image_id, img, label = super(InstanceDataset, self)._load_data(idx)
        instance_label_path = os.path.join(self.instance_dataset_path, image_id)
        instance_label = np.asarray(imread(instance_label_path))
        label = np.dstack((label, instance_label))
        return image_id, img, label

    def __getitem__(self, index):
        data = super(InstanceDataset, self).__getitem__(index)
        label, instance_label = data['label'][...,0], data['label'][...,1]
        instance_label[instance_label == 255] = 0
        # if np.any(label == 1):
        #     cv2.imwrite(os.path.join(self.vis_dir, 'label.png'), label)
        #     cv2.imwrite(os.path.join(self.vis_dir, 'instance_label.png'), instance_label)
        #     pdb.set_trace()
        data.update(label=label, instance_label=instance_label)
        return data

@register.attach('angle')
class AngleDataset(data.Dataset):
    """
    Base dataset class
    """
    ignore_label = -100
    min_angle = -35
    max_angle = 125
    bin_len = 10
    n_bins = int((max_angle - min_angle) / bin_len)
    bin_edges = np.arange(min_angle, max_angle + bin_len, bin_len)
    bin_tresh = bin_len / 2.5

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
        self.augmentations = Compose(augmentations)
        self.masks_test = masks_test

    def tile_mask(self, mask, dim=2):
        repeats = [1] * 3
        repeats[2] = dim
        return np.tile(mask[...,np.newaxis], repeats)

    def compute_orientations_gt(self, label, angles, img=None):

        kernel = np.ones((3,3), np.uint8)
        iterations = 5
        eroded_label = cv2.erode((label == 1).astype(np.uint8), kernel, iterations=iterations)
        # plt.figure()
        # plt.imshow(label)
        # plt.figure()
        # plt.imshow(eroded_label)
        # plt.show()

        def _compute_gt(ori_gt, angle_range):
            

            _, lines_angles, lines_dists = lines.search_lines(eroded_label, angle_range)
            lines_angles_deg = np.rad2deg(lines_angles).tolist()
            indices = np.digitize(lines_angles_deg, self.bin_edges).tolist()
            detected_lines = lines.get_lines(lines_dists, lines_angles)

            for _line, _angle, idx  in zip(detected_lines, lines_angles_deg, indices):
                line_mask = np.logical_and(lines.create_grid(label.shape, [_line], width=16) == 1, (label != 255))
                digitized_angle = (self.bin_edges[idx] + self.bin_edges[idx-1]) / 2
                angle_error = _angle - digitized_angle
                if abs(angle_error) > self.bin_tresh:
                    other_idx = idx + (angle_error > 0) - (angle_error < 0)
                    ori_gt[..., other_idx][line_mask] = 1.0
                ori_gt[..., idx][line_mask] = 1.0
                ori_gt[...,-1][line_mask] = 0.0

            # common_indices = np.unique(indices).tolist()
            # for common_idx in common_indices:
            #     print(np.sum(ori_gt[...,common_idx]))
            #     plt.figure()
            #     plt.imshow(ori_gt[...,common_idx])

            return ori_gt

        ori_gt = np.zeros(label.shape + (self.n_bins,), dtype=np.float32)
        ori_gt = np.dstack((ori_gt, np.ones(label.shape + (1,), dtype=np.float32)))

        if np.any(label) and angles is not None:

            angle_h, angle_v = angles
            angle_range_h = (angle_h - 2.5, angle_h + 2.5)
            angle_range_v = (angle_v - 2.5, angle_v + 2.5)

            ori_gt = _compute_gt(ori_gt, angle_range_h)
            ori_gt = _compute_gt(ori_gt, angle_range_v)

        if img is not None:
            _mask = (ori_gt[...,-1].astype(int) == 0).astype(np.uint8)
            # _mask2 = label.astype(np.uint8)
            vis_img = vis.vis_seg(img, _mask, vis.make_palette(2))
            # vis_img2 = vis.vis_seg(img, _mask2, vis.make_palette(2))
            plt.figure()
            plt.imshow(vis_img)
            # plt.figure()
            # plt.imshow(vis_img2)
            plt.show()

        ori_gt = np.transpose(ori_gt, (2, 0, 1))

        return ori_gt


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
        # pdb.set_trace()
        # plt.imshow(label)
        # plt.show()
        image, label, angles = self.augmentations(image, label, angles)
        # pdb.set_trace()
        # plt.imshow(label)
        # plt.show()

        label = label.astype(np.int64)

        mask = (label != 255)
        label_3c = label.copy()
        label_3c[label == 0] = 1
        label_3c[mask] -= 1

        label_2c = label.copy()
        label_2c[np.logical_or(label == 2, label == 3)] = 255

        label_3c[label_3c == 255] = self.ignore_label
        label_2c[label_2c == 255] = self.ignore_label

        ori_gt = self.compute_orientations_gt(label_2c, angles)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        return dict(image_id=image_id, 
                    image=image, 
                    label=label, 
                    ori_gt=ori_gt,
                    label_3c=label_3c,
                    label_2c=label_2c)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


@register.attach('angle_v2')
class AngleDataset_v2(data.Dataset):
    """
    Base dataset class
    """
    ignore_label = -100
    min_angle = -45
    max_angle = 45
    bin_len = 7.5
    n_bins = int((max_angle - min_angle) / bin_len)
    bin_edges = np.arange(min_angle, max_angle + bin_len, bin_len)
    bin_tresh = bin_len / 2.5

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
        self.augmentations = Compose(augmentations)
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

        ori_gt = np.zeros(self.n_bins, dtype=np.float32)
        if angles is not None:

            bin_idx = np.digitize(angles[1], self.bin_edges)
            if bin_idx == self.n_bins:
                ori_gt[self.n_bins] = 1.0
            elif bin_idx == 0:
                ori_gt[0] = 1.0
            else:
                ori_gt[bin_idx] = 1.0
                bin_center = (self.bin_edges[bin_idx] - self.bin_edges[bin_idx-1]) / 2
                angle_error = angles[1] - bin_center
                if abs(angle_error) > self.bin_tresh:
                    other_idx = bin_idx + (angle_error > 0) - (angle_error < 0)
                    ori_gt[other_idx] = 1.0

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
                    ori_gt=ori_gt,
                    label_3c=label_3c,
                    label_2c=label_2c)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


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
        self.augmentations = Compose(augmentations)
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





