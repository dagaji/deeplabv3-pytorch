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

class LinesDatasetFactory:

    bbox_size = (833, 449)
    M = 50

    def norm_coords(self, coords):
        x_norm = float(self.bbox_size[0] - 1) / 2
        y_norm = float(self.bbox_size[1] - 1) / 2
        coords[:, 0] = coords[:,0] / x_norm - 1
        coords[:, 1] = coords[:, 1] / y_norm - 1
        return coords

    def line_coords(self, intersect_points, orientation, n_points):

        step_len = distance.euclidean(intersect_points[0], intersect_points[1]) / n_points
        orientation_rad = np.deg2rad(orientation)
        unit_vector = np.array((np.cos(orientation_rad), np.sin(orientation_rad)))
        line_points = []
        for i in range(1, n_points):
            line_point = np.array(intersect_points[0]) + i * step_len * unit_vector
            line_points.append(line_point)
        line_points = np.array(line_point)
        return self.norm_coords(line_points)


    def compute_anchors_grid(self):

        anchors_v_intersect = lines.get_anchors(0.0, 50, self.bbox_size[::-1], is_vertical=True)
        anchors_h_intersect = lines.get_anchors(90.0, 25, self.bbox_size[::-1], is_vertical=False)

        grids_v = []
        for intersect in anchors_v_intersect:
            grid = self.line_coords(intersect, 0.0, M).reshape(M,1,2)
            grids_v.append(grid)

        grids_h = []
        for intersect in anchors_h_intersect:
            grid = self.line_coords(intersect, 90.0, 2 * M).reshape(2* M,1,2)
            grids_h.append(grid)

        return grids_v, grids_h


    def __init__(self, root):

        with open(os.path.join(root, 'lines_gt.pickle'), "rb") as file:
            self.lines_gt = pickle.load(file)

        self.grids_v, self.grids_h = self.compute_anchors_grid()

    def get_line_dataset(self, image_id):
        lines_gt_v, lines_gt_h = self.lines_gt[image_id]['v'], self.lines_gt[image_id]['h']
        return LinesDataset(self.grids_v + self.grids_h, np.concatenate(lines_gt_v,lines_gt_h))


class LinesDataset(data.Dataset):

    def __init__(self, grids, gts):
        self.grids = grids
        self.gts = gts

    def __getitem__(self, index):
        grid = self.grids[idx]
        gt = self.gts[idx]
        return grid, gt

    def __len__(self):
        return len(self.grids)



class RotatedImagesDataset(data.Dataset):

    
    def __init__(self, root, id_list_path):

        
        self.root = root
        id_list = np.loadtxt(id_list_path, dtype=str)
        self.id_list = []
        for _id in id_list:
            if "APR" in _id:
                self.id_list.append(_id)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]

    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx] + '.png'
        img_path = os.path.join(self.root, "images", image_id)
        img = np.asarray(imread(img_path))
        return image_id, img

    def __getitem__(self, index):

        image_id, image, lines_labels = self._load_data(index)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()

        return dict(image_id=image_id, image=image)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


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



# @register.attach('dataset_multi')
# class DatasetMulti(BaseDataset):

#     def __init__(self, root, id_list_path, augmentations=[], scaling_factor=0.5):
#         super(DatasetMulti, self).__init__(root, id_list_path, augmentations)
#         self.scaling_factor = scaling_factor

#     def __getitem__(self, index):
#         data = super(DatasetMulti, self).__getitem__(index)
#         img, label = data['image'], data['label']
#         img_down = scale_img(img, self.scaling_factor, is_mask=False)
#         label_down = scale_img(label, self.scaling_factor, is_mask=True)
#         data.update(img_down=img_down, label_down=label_down)
#         return data

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

        # label_3c_aux = label_3c.copy()
        # label_3c_aux[label_3c == 255] = 3
        # label_2c_aux = label_2c.copy()
        # label_2c_aux[label_2c == 255] = 2
        # plt.figure()
        # plt.imshow(label_3c_aux)
        # plt.figure()
        # plt.imshow(label_2c_aux)
        # plt.show()

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


def scale_img(img, scaling_factor, is_mask=False):
    if is_mask:
        img_down = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
    else:
        img_down = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img_down