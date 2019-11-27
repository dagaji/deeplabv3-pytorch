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
        self.line_sampler = lines.LineSampler(angle_step=5.0)
        self.max_offset = 45
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

        def _get_iou(true_lines, true_masks, proposed_line):

            opt_lines = []
            ious = []
            proposed_rho, proposed_theta = proposed_line

            for idx in np.arange(true_masks.shape[0]):

                _true_line_coeffs = lines.general_form(*true_lines[idx].tolist())
                intersect_borders = np.array(lines.find_intesect_borders(_true_line_coeffs, sz))
                mid_point = intersect_borders.mean(0)
                opt_rho = mid_point[0] * np.cos(proposed_theta) + mid_point[1] * np.sin(proposed_theta)
                opt_line = (opt_rho, proposed_theta)

                proposed_mask = _create_mask(opt_line, guard=False, width=16)
                not_ignored = np.logical_and(true_masks[idx] > 0, true_masks[idx] < 2)
                hist = np.bincount(2 * true_masks[idx][not_ignored].flatten() + proposed_mask[not_ignored].flatten(), minlength=4).reshape((2,2))
                iou = (np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[1]
                
                ious.append(iou)
                opt_lines.append(opt_line)

            opt_rhos = np.array(opt_lines)[:,0]
            _idx = np.argmin(np.abs(opt_rhos - proposed_rho))

            return opt_lines[_idx], ious[_idx], true_masks[_idx]

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
        iou_gt = np.zeros(n_lines, dtype=np.float32)
        reg_gt = np.zeros(n_lines, dtype=np.float32)

        for idx in np.arange(n_lines):
            distance = np.abs(proposed_lines[idx] - true_lines)
            close_lines = np.logical_and(distance[:,0] < self.max_offset, distance[:,1] < np.deg2rad(5.0))
            if np.any(close_lines):
                opt_line, iou, _true_mask = _get_iou(true_lines[close_lines], true_lines_masks[close_lines], proposed_lines[idx])
                lines_gt[idx] = 1.0
                iou_gt[idx] = iou
                reg_gt[idx] = (proposed_lines[idx][0] - opt_line[0]) / self.max_offset

        return lines_gt, iou_gt, reg_gt


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
            angle_range_v = self.rot_angles[np.argsort(angle_dist)[:2]]
            angle_range_v.sort()
            angle_range_h = angle_range_v + 90.0

            proposed_lines_v, intersects_points_v = self.line_sampler(angle_range_v, label_test.shape)
            proposed_lines_h, intersects_points_h = self.line_sampler(angle_range_h, label_test.shape)

            lines_gt_v, iou_gt_v, reg_gt_v = self.get_line_gt(true_lines_v, proposed_lines_v, label_test.shape)
            lines_gt_h, iou_gt_h, reg_gt_h = self.get_line_gt(true_lines_h, proposed_lines_h, label_test.shape)

            lines_gt = np.append(lines_gt_v, lines_gt_h)
            iou_gt = np.append(iou_gt_v, iou_gt_h)
            reg_gt = np.append(reg_gt_v, reg_gt_h)
            proposed_lines = np.array(proposed_lines_v + proposed_lines_h, dtype=np.float32)
            intersects_points = np.array(intersects_points_v + intersects_points_h, dtype=np.float32)
            
        return dict(image_id=image_id, image=image, seg_label=seg_label, intersects_points=intersects_points, lines_gt=lines_gt, iou_gt=iou_gt, reg_gt=reg_gt)

    def __len__(self):
        return len(self.id_list)

    def __repr__(self):
        fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


@register.attach('roi_dataset')
class ROIDataset(data.Dataset):

    def __init__(self, root, id_list_path, angle_step=15.0, min_angle=-30.0, max_angle=30.0, augmentations=[]):
        self.root = root
        self.id_list = np.loadtxt(id_list_path, dtype=str)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]
        self.augmentations = Compose(augmentations)
        self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.ROI_sampler = lines.ROISampler(angle_step=1.0)
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

    def get_line_gt(self, true_lines, ROI_projections):

        n_lines = len(ROI_projections)
        lines_gt = np.zeros(n_lines, dtype=np.float32)
        reg_gt = np.zeros((n_lines, 2), dtype=np.float32)

        for idx, _ROI_projection in enumerate(ROI_projections):
            for _true_line in true_lines:
                _projection = _ROI_projection(lines.general_form(*_true_line))
                if _projection is not None:
                    lines_gt[idx] = 1.0
                    reg_gt[idx] = np.array(_projection, dtype=np.float32)
                    break

        return lines_gt, reg_gt

    def plot_ROIs(self, true_lines, ROIs_limits, line_gt, sz):

        ROIs_limits1, ROIs_limits2 = np.array(ROIs_limits).transpose((1,0,2))
        is_positive = line_gt.astype(bool)

        true_mask = lines.create_grid(sz, true_lines)
        all_lines = lines.create_grid(sz, ROIs_limits1.tolist() + ROIs_limits2.tolist())
        all_lines[true_mask == 1] = 2

        plt.figure()
        plt.imshow(all_lines)
        plt.show()

        # ROIs_limits_positive = ROIs_limits1[is_positive].tolist() + ROIs_limits2[is_positive].tolist()
        # ROIs_limits_negative = ROIs_limits1[~is_positive].tolist() + ROIs_limits2[~is_positive].tolist()

        # true_mask = lines.create_grid(sz, true_lines)

        # positive_mask = true_mask.copy()
        # positive_mask[lines.create_grid(sz, ROIs_limits_positive) == 1] = 2

        # negative_mask = true_mask.copy()
        # negative_mask[lines.create_grid(sz, ROIs_limits_negative) == 1] = 2

        # plt.figure()
        # plt.imshow(positive_mask)
        # plt.figure()
        # plt.imshow(negative_mask)
        # plt.show()


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
            print(angle_dist.min())
            closest_angle = self.rot_angles[np.argmin(angle_dist)]

            angle_range_v = np.array((closest_angle, closest_angle))
            angle_range_h = angle_range_v + 90.0

            ROIs_projection_v, ROIs_coords_v, ROIs_limits_v = self.ROI_sampler(angle_range_v, label_test.shape, npoints=70)
            ROIs_projection_h, ROIs_coords_h, ROIs_limits_h = self.ROI_sampler(angle_range_h, label_test.shape, npoints=70)

            lines_gt_v, reg_gt_v = self.get_line_gt(true_lines_v, ROIs_projection_v)
            lines_gt_h, reg_gt_h = self.get_line_gt(true_lines_h, ROIs_projection_h)

            lines_gt = np.append(lines_gt_v, lines_gt_h)
            reg_gt = np.vstack((reg_gt_v, reg_gt_h))
            ROIs_coords = np.array(ROIs_coords_v + ROIs_coords_h, dtype=np.float32)

            self.plot_ROIs(true_lines_v, ROIs_limits_v, lines_gt_v, label_test.shape)
            self.plot_ROIs(true_lines_h, ROIs_limits_h, lines_gt_h, label_test.shape)

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
            

        return dict(image_id=image_id, image=image, seg_label=seg_label, ROIs_coords=ROIs_coords, lines_gt=lines_gt, reg_gt=reg_gt)

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


