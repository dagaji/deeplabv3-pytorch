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

def string2msec(time_string):
    time_min = int(time_string.split(':')[0])
    time_sec = int(time_string.split(':')[1])
    time_sec += time_min * 60
    time_msec = 1000 * time_sec
    return time_msec


def msec2string(time_msec):
    time_sec = time_msec // 1000
    time_min = time_sec // 60
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
class VideoDataset(data.Dataset):

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


@register.attach('hist_dataset')
class HistDataset(data.Dataset):

    def __init__(self, root, id_list_path, angle_step=15.0, min_angle=-30.0, max_angle=30.0, augmentations=[]):
        self.root = root
        self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = Compose(augmentations)
        self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.line_sampler = lines.LineSampler(angle_step=1.0, rho_step=50)
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

        def _get_diff_intersects(proposed_line, true_lines):

            proposed_line_coeffs = lines.general_form(*proposed_line)
            _intersects = lines.find_intesect_borders(proposed_line_coeffs, sz)
            proposed_line_intersects = np.array(_intersects[0] + _intersects[1], dtype=np.float32)[np.newaxis,:]

            true_lines_intersects = []
            for true_line in true_lines.tolist():
                true_line_coeffs = lines.general_form(*true_line)
                _intersects = lines.find_intesect_borders(true_line_coeffs, sz)
                true_lines_intersects.append(_intersects[0] + _intersects[1])
            true_lines_intersects = np.array(true_lines_intersects, dtype=np.float32)

            diff_intersects = true_lines_intersects - proposed_line_intersects
            diff_intersects_l1 = np.abs(diff_intersects).sum(1)

            return diff_intersects[np.argmin(diff_intersects_l1)]

        true_lines = np.array(true_lines, dtype=np.float32)
        proposed_lines = np.array(proposed_lines, dtype=np.float32)

        n_lines = proposed_lines.shape[0]
        lines_gt = np.zeros(n_lines, dtype=np.float32)
        reg_gt = np.zeros((n_lines, 4), dtype=np.float32)

        for idx in np.arange(n_lines):
            distance = np.abs(proposed_lines[idx] - true_lines)
            close_lines = np.logical_and(distance[:,0] < 25, distance[:,1] < np.deg2rad(5.0))
            if np.any(close_lines):
                lines_gt[idx] = 1.0
                # proposed_line_coeffs = lines.general_form(*proposed_lines[idx])
                # closest_line_coeffs = lines.general_form(*true_lines[np.argmin(distance)])
                # proposed_line_intersects = lines.find_intesect_borders(proposed_line_coeffs, sz)
                # closest_line_intersects = lines.find_intesect_borders(closest_line_coeffs, sz)
                # offset = np.array(proposed_line_intersects) - np.array(closest_line_intersects)
                # reg_gt[idx] = np.hstack((offset[0], offset[1]))
                reg_gt[idx] = _get_diff_intersects(proposed_lines[idx], true_lines[close_lines])

        return lines_gt, reg_gt / 500.0


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
            
            angle_dist = np.abs(self.rot_angles - np.rad2deg(angles_v).mean())
            closest_angle = self.rot_angles[np.argmin(angle_dist)]

            angle_range_v = np.array((closest_angle, closest_angle))
            angle_range_h = angle_range_v + 90.0

            sampled_points_v, proposed_lines_v, _ = self.line_sampler(angle_range_v, label_test.shape, npoints=100)
            sampled_points_h, proposed_lines_h, _ = self.line_sampler(angle_range_h, label_test.shape, npoints=100)

            lines_gt_v, reg_gt_v = self.get_line_gt(true_lines_v, proposed_lines_v, label_test.shape)
            lines_gt_h, reg_gt_h = self.get_line_gt(true_lines_h, proposed_lines_h, label_test.shape)

            lines_gt = np.append(lines_gt_v, lines_gt_h).astype(bool)
            reg_gt = np.vstack((reg_gt_v, reg_gt_h))[lines_gt]
            proposed_lines_positive = np.array(proposed_lines_v + proposed_lines_h, dtype=np.float32)[lines_gt]
            sampled_points_positive = np.stack(sampled_points_v + sampled_points_h)[lines_gt]
            n_positives = proposed_lines_positive.shape[0]

            ####################################################################################################

            negative_angle = closest_angle + 20.0
            angle_range_v = np.array((negative_angle, negative_angle))
            angle_range_h = angle_range_v + 90.0

            sampled_points_v, proposed_lines_v, _ = self.line_sampler(angle_range_v, label_test.shape, npoints=100)
            sampled_points_h, proposed_lines_h, _ = self.line_sampler(angle_range_h, label_test.shape, npoints=100)
            proposed_lines_negative = np.array(proposed_lines_v + proposed_lines_h, dtype=np.float32)
            sampled_points_negative = np.stack(sampled_points_v + sampled_points_h)
            shuffled_indices = np.random.permutation(proposed_lines_negative.shape[0])[:n_positives]
            proposed_lines_negative = proposed_lines_negative[shuffled_indices]
            sampled_points_negative = sampled_points_negative[shuffled_indices]
            
            lines_gt = np.append(np.ones(n_positives, dtype=np.float32), np.zeros(n_positives, dtype=np.float32))
            reg_gt = np.vstack((reg_gt, np.zeros((n_positives, 4), dtype=np.float32)))
            sampled_points = np.concatenate((sampled_points_positive, sampled_points_negative), axis=0)

            # aux_mask1 = lines.create_grid(label_test.shape, proposed_lines_positive.tolist())
            # aux_mask2 = lines.create_grid(label_test.shape, proposed_lines_negative.tolist())
            # aux_mask3 = lines.create_grid(label_test.shape, true_lines_v + true_lines_h)
            # plt.figure()
            # plt.imshow(aux_mask1)
            # plt.figure()
            # plt.imshow(aux_mask2)
            # plt.figure()
            # plt.imshow(aux_mask3)
            # plt.show()

        return dict(image_id=image_id, image=image, seg_label=seg_label, line_points=sampled_points, lines_gt=lines_gt, reg_gt=reg_gt)

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
        # descomentar cuando se copie imagen con imwrite
        #label = np.asarray(imread(label_path))[..., 0]
        label = np.asarray(imread(label_path))
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

@register.attach('pano_dataset')
class PanoDataset(BaseDataset):

    def __init__(self, root, id_list_path, augmentations=[], change_ignore_index=False, train=True):
        super(PanoDataset, self).__init__(root, 
                                          id_list_path, 
                                          augmentations=augmentations, 
                                          masks_test=True, 
                                          change_ignore_index=change_ignore_index)


    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.id_list[idx]
        img_path = os.path.join(self.root, "images", image_id)
        img = np.asarray(imread(img_path))
    
        return image_id, img


    def __getitem__(self, index):

        image_id, image = self._load_data(index)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.var)

        return dict(image_id=image_id, image=image)



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


@register.attach('mosaic_dataset')
class MosaicDataset(data.Dataset):

    delay_msec = int(1000)
    _params_path = os.path.join("/home/davidgj/projects/refactor", "APR_TAX_RWY_panos_2", "{}", "parameters", "offset.p")

    def __init__(self, mosaic_root, img_root, id_list_path, augmentations=[], resize_scale=0.25):
        
        self.mosaic_root = mosaic_root
        self.img_root = img_root
        self.mosaic_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = Compose_v2(augmentations)
        self.setup()
        self.resize_scale = resize_scale

    def get_mid_time_frame(self, mosaic_id):

        video_name, interval_string = mosaic_id.split("_")
        end_time, start_time = interval_string.split("-")
        nframes = (string2msec(end_time) - string2msec(start_time)) // self.delay_msec + 1
        mid_time = msec2string(string2msec(start_time) + (nframes // 2 + nframes % 2 - 1) * self.delay_msec)
        return "{}_{}".format(video_name, mid_time)

    def setup(self,):

        imgs_dir = os.path.join(self.img_root, "images")
        imgs_name_list = [glob.parts[-1].split('.')[0] for glob in Path(imgs_dir).glob("*.png")]
        new_mosaic_list = []
        for mosaic_name in self.mosaic_list.tolist():
            mid_time_frame = self.get_mid_time_frame(mosaic_name)
            if mid_time_frame in imgs_name_list:
                new_mosaic_list.append(mosaic_name)

        self.mosaic_list = np.array(new_mosaic_list)

        video_names = list(set([mosaic_name.split("_")[0] for mosaic_name in self.mosaic_list]))

        self.params = {}
        for video_name in video_names:
            params_path = self._params_path.format(video_name)
            with open(params_path, 'rb') as handle:
                video_params = pickle.load(handle, encoding='latin1')
            self.params[video_name] = video_params


    def __len__(self):
        return len(self.mosaic_list)


    def _load_data(self, idx):
        """
        Load the image and label in numpy.ndarray
        """

        def _load_img_label(img_path, label_path):
            img = np.asarray(imread(img_path))
            label = np.asarray(imread(label_path))
            return img, label

        mosaic_id = self.mosaic_list[idx]

        video_name, interval_string = mosaic_id.split("_")
        mid_time = list(self.params[video_name][interval_string].keys())[0]
        frame_id = video_name + "_" + mid_time

        mosaic_img_path = os.path.join(self.mosaic_root, "images", mosaic_id + ".png")
        mosaic_label_path = os.path.join(self.mosaic_root, "masks", mosaic_id + ".png")
        mosaic_img, mosaic_label = _load_img_label(mosaic_img_path, mosaic_label_path)

        frame_img_path = os.path.join(self.img_root, "images", frame_id + ".png")
        frame_label_path = os.path.join(self.img_root, "masks", frame_id + ".png")
        frame_img, frame_label = _load_img_label(frame_img_path, frame_label_path)
        frame_label = frame_label[..., 0]

        grid_coords = self.params[video_name][interval_string][mid_time]
        grid_coords = grid_coords.transpose((1,2,0))

        return mosaic_id + ".png", frame_id + ".png", mosaic_img, frame_img, frame_label, grid_coords


    def __getitem__(self, index):

        def _normalize_image(image):
            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.var)
            image = image.numpy()
            return image

        def _normalize_grid(grid_coords, sz):
            H, W = sz
            grid_coords[..., 0] = grid_coords[..., 0] / (W / 2) - 1.0
            grid_coords[..., 1] = grid_coords[..., 1] / (H / 2) - 1.0
            return grid_coords

        _, frame_id, mosaic_img, frame_img, frame_label, grid_coords = self._load_data(index)

        frame_img, (frame_label, grid_coords) = self.augmentations(frame_img, frame_label, grid_coords)
        grid_coords = _normalize_grid(grid_coords, mosaic_img.shape[:2])
        # print("mosaic_size:{}".format(mosaic_img.shape))
        mosaic_img = cv2.resize(mosaic_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        # print("mosaic_size:{}".format(mosaic_img.shape))
        mosaic_img = _normalize_image(mosaic_img)
        frame_img = _normalize_image(frame_img)

        frame_label = frame_label.astype(np.int64)

        return dict(frame_id=frame_id, frame_img=frame_img, mosaic_img=mosaic_img, grid_coords=grid_coords, frame_label=frame_label)


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


