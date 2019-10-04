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
from scipy import ndimage

def string2msec(time_string):
    time_min = int(time_string.split(':')[0])
    time_sec = int(time_string.split(':')[1])
    time_sec += time_min * 60
    time_msec = 1000 * time_sec
    return time_msec


def msec2string(time_msec):
    time_sec = time_msec / 1000
    time_min = time_sec / 60
    time_string = "{}:{:02d}".format(time_min, time_sec - time_min * 60)
    return time_string

def downscale(img, max_dim):

    height, width = img.shape[:2]

    if max_dim < height or max_dim < width:
        scaling_factor = min(max_dim / float(width), max_dim / float(height))
        img_down = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return img_down
    else:
        return None

def undistort(img, dist, mtx):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst

class VideoLoader:

    def __init__(self, video_path, camera_name, max_dim=1000):

        self.vidcap = cv2.VideoCapture(video_path)
        camera_dir = os.path.join("calib_data", camera_name)
        self.dist = np.load(os.path.join(camera_dir, 'dist.npy'))
        self.mtx = np.load(os.path.join(camera_dir, 'mtx.npy'))
        self.max_dim = max_dim

    def frame_at(self, time_msec):

        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, time_msec)
        success, frame = self.vidcap.read()
        if success:
            frame = self.process_frame(frame)
            return frame
        else:
            return None

    def process_frame(self, frame):
        frame = undistort(frame, self.dist, self.mtx)
        frame = downscale(frame, self.max_dim)
        return frame


@register.attach('video_dataset')
class BaseDataset(data.Dataset):

    def __init__(self, video_path, camera_name, start_time, end_time, fps=10.0):
        self.video_loader = VideoLoader(video_path, camera_name)
        start_time_msec = string2msec(start_time)
        end_time_msec = string2msec(end_time)
        step_msec = int(1000 / fps)
        self.time_msec_array = np.arange(start_time_msec, end_time_msec + step_msec, step_msec)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.time_msec_array)

    def __getitem__(self, index):

        time_msec = self.time_msec_array[index]
        frame = self.video_loader.frame_at(time_msec)
        image = TF.to_tensor(frame[...,::-1].copy())
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()
        return image, frame.astype(np.int64)

@register.attach('pano_dataset')
class BaseDataset(BaseDataset):

    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx] + '.png'
        img_path = os.path.join(self.root, "images", image_id)
    
        return image_id, img


    def __getitem__(self, index):

        image_id, image = self._load_data(index)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)

        return dict(image_id=image_id, image=image)





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


@register.attach('edges')
class EdgesDataset(MultiTaskDataset):


    def __init__(self, root, id_list_path, augmentations=[], masks_test=False, change_ignore_index=False):
        super(EdgesDataset, self).__init__(root, id_list_path, augmentations, masks_test=False, change_ignore_index=False)

    def __getitem__(self, index):
        data = super(EdgesDataset, self).__getitem__(index)
        label_3c = data['label_3c']

        sx = ndimage.sobel(label_3c, axis=0, mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(label_3c, axis=1, mode='constant')
        # Get square root of sum of squares
        edges_label = (np.hypot(sx,sy) > 0).astype(np.uint8)

        ignore_mask = label_3c.copy()
        ignore_mask[label_3c != self.ignore_label] = 0
        ignore_mask[label_3c == self.ignore_label] = 1
        ignore_mask = ignore_mask.astype(np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, np.ones((7,7), np.uint8), iterations=3)
        
        edges_label[ignore_mask == 1] = 0
        h, w = edges_label.shape
        edges_label[0,:] = 0
        edges_label[:,0] = 0
        edges_label[h-1,:] = 0
        edges_label[:,w-1] = 0
        edges_label = cv2.dilate(edges_label, np.ones((5,5), np.uint8), iterations=2)
        # label_3c_aux = label_3c.copy()
        # label_3c_aux[label_3c == self.ignore_label] = 3
        # plt.figure()
        # plt.imshow(label_3c_aux)
        # plt.figure()
        # plt.imshow(edges_label)
        # plt.show()
        edges_label = edges_label.astype(np.int64)
        edges_label[ignore_mask == 1] = self.ignore_label

        data.update(edges_label=edges_label)
        return data


