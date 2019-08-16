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
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import deeplabv3.lines as lines
import pickle
from scipy.spatial import distance

class LinesDatasetFactory:

    bbox_size = (833, 449)
    M = 50

    def norm_coords(self, coords):
        x_norm = float(self.bbox_size[0] - 1) / 2
        y_norm = float(self.bbox_size[1] - 1) / 2
        coords[:, 0] = coords[:,0] / x_norm - 1
        coords[:, 1] = coords[:, 1] / y_norm - 1
        coords = coords.astype(np.float32)
        return coords

    def line_coords(self, intersect_points, orientation, n_points, plot=False):

        if plot:
            fig, ax = plt.subplots(1)
            ax.imshow(np.zeros(self.bbox_size[::-1] + (3,), dtype=np.uint8))

        step_len = distance.euclidean(intersect_points[0], intersect_points[1]) / n_points
        orientation_rad = np.deg2rad(orientation)
        unit_vector = np.array((np.sin(orientation_rad), np.cos(orientation_rad)))
        line_points = []
        for i in range(1, n_points):
            line_point = np.array(intersect_points[0]) + i * step_len * unit_vector
            line_points.append(line_point)
            if plot:
                circle = plt.Circle(tuple(line_point.tolist()), 2, color='b')
                ax.add_patch(circle)

        if plot:
            plt.show()
        line_points = np.array(line_points)
        return self.norm_coords(line_points)


    def compute_anchors_grid(self):

        anchors_v_intersect = lines.get_anchors(0.0, 50, self.bbox_size[::-1], is_vertical=True)
        anchors_h_intersect = lines.get_anchors(90.0, 25, self.bbox_size[::-1], is_vertical=False)

        grids_v = []
        for intersect in anchors_v_intersect:
            grid = self.line_coords(intersect, 0.0, self.M)
            grid = grid.reshape(grid.shape[0],1,2)
            grids_v.append(grid)

        grids_h = []
        for intersect in anchors_h_intersect:
            grid = self.line_coords(intersect, 90.0, 2 * self.M)
            grid = grid.reshape(grid.shape[0],1,2)
            grids_h.append(grid)

        return grids_v, grids_h


    def __init__(self, root):

        with open(os.path.join(root, 'lines_gt.pickle'), "rb") as file:
            self.lines_gt = pickle.load(file)

        self.grids_v, self.grids_h = self.compute_anchors_grid()

    def get_line_datasets(self, image_id):
        lines_gt_v, lines_gt_h = self.lines_gt[image_id]['v'], self.lines_gt[image_id]['h']
        return LinesDataset(self.grids_v, lines_gt_v), LinesDataset(self.grids_h, lines_gt_h)


class LinesDataset(data.Dataset):

    def __init__(self, grids, gts):
        self.grids = grids
        self.gts = gts

    def __getitem__(self, index):
        grid = self.grids[index]
        gt = self.gts[index]
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

        image_id, image = self._load_data(index)
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