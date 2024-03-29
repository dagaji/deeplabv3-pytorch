import matplotlib
matplotlib.use('tkagg')
import numpy as np
from .register import register
import cv2
from scipy import stats
import pdb
import matplotlib.pyplot as plt
import deeplabv3.vis as vis

def _get_augmentation(aug, params):
    Augmentation = register.get(aug)
    if isinstance(params, dict):
        return Augmentation(**params)
    elif isinstance(params, list):
        return Augmentation(params)
    return Augmentation()

def _get_random_gen(sigma, alfa=1.5):
        lower, upper = -alfa * sigma, alfa * sigma
        mu, sigma = 0, sigma
        trunc_gen = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        return trunc_gen

class Compose(object):

    def __init__(self, augmentations_list):
        self.augmentations = []
        for el in augmentations_list:
            aug_name = str(*el.keys())
            params = el[aug_name]
            self.augmentations.append(_get_augmentation(aug_name, params))

    def __call__(self, img, mask, angles=None):
        for aug in self.augmentations:
            img, mask, angles = aug(img, mask, angles)
        return img, mask, angles


@register.attach('random_crop_angle')
class _RandomCrop(object):

    def __init__(self, w, h):
        self.bbox_size = (w, h)

    def crop(self, img, top_border):
        x_limits = (top_border[0], top_border[0] + self.bbox_size[0])
        y_limits = (top_border[1], top_border[1] + self.bbox_size[1])
        bbox = img[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]
        return bbox

    def __call__(self, img, mask, angles):
        h, w = img.shape[:2]
        x_max = w - self.bbox_size[0]
        y_max = h - self.bbox_size[1]
        top_border_x = np.random.uniform(0, x_max, 1).astype(int)
        top_border_y = np.random.uniform(0, y_max, 1).astype(int)
        top_border = (top_border_x[0], top_border_y[0])
        cropped_img = self.crop(img, top_border)
        cropped_mask = self.crop(mask, top_border)
        return cropped_img, cropped_mask, angles


@register.attach('random_flip_angle')
class _RandomFlip(object):

    def __call__(self, img, mask, angles):
        flip_op = np.random.choice([-1,0,1,2])
        if flip_op != 2:
            img = cv2.flip(img, flip_op)
            mask = cv2.flip(mask, flip_op)
            if flip_op != -1 and angles is not None:
                angles = (180 - angles[0], 180 - angles[1])
        return img, mask, angles
        

@register.attach('rotate_angle')
class _RandomRotation(object):

    def __init__(self, sigma_angle, w, h, aux_index=100, ignore_index=255):
        self.angle_gen = _get_random_gen(sigma_angle)
        self.crop_size = (w, h)
        self.aux_index = aux_index
        self.ignore_index = ignore_index

    def __call__(self, img, label, angles):

        angle = self.angle_gen.rvs()
        img = self.rotate_img(img, angle, self.crop_size)
        label = self.rotate_img(label, angle, self.crop_size, is_mask=True)
        if angles is not None:
            angles = (angles[0] - angle, angles[1] - angle)

        ignore_mask = label.copy()
        ignore_mask[label != self.aux_index] = 0
        kernel = np.ones((21,21), np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel, iterations=3)
        label[ignore_mask == self.aux_index] = self.ignore_index

        return img, label, angles

    def rotate_img(self, mat, angle, bbox_size, is_mask=False):

        height, width = mat.shape[:2]
        image_center = (width/2, height/2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        abs_cos = abs(rotation_mat[0,0])
        abs_sin = abs(rotation_mat[0,1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        bound_size = (bound_w, bound_h)

        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        if is_mask:
            rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_NEAREST, borderValue=self.aux_index)
        else:
            rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_AREA, borderValue=0)

        rotated_center = (bound_w/2, bound_h/2)

        x_limits = (np.array((-bbox_size[0]/2, bbox_size[0]/2 + bbox_size[0]%2 - 1)) + rotated_center[0]).astype(int)
        y_limits = (np.array((-bbox_size[1]/2, bbox_size[1]/2 + bbox_size[1]%2 - 1)) + rotated_center[1]).astype(int)

        bbox = rotated_mat[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]

        return bbox


@register.attach('choice_angle')
class _RandomChoice(object):

    def __init__(self, augmentations_list):
        self.augmentations = []
        for el in augmentations_list:
            aug_name = str(*el.keys())
            params = el[aug_name]
            self.augmentations.append(_get_augmentation(aug_name, params))

    def __call__(self, img, mask, angles):
        if len(self.augmentations):
            indices = range(len(self.augmentations))
            idx = np.random.choice(indices)
            img, mask, angles = self.augmentations[idx](img, mask, angles)
        return img, mask, angles


@register.attach('pca_angle')
class _PCAColorAugmentation(object):

    def __init__(self, sigma_pca):
        self.perturb_gen = _get_random_gen(sigma_pca, alfa=1.0)

    def __call__(self, img, mask, angles):

        perturb = self.perturb_gen.rvs((3,3))
        img = img.astype(float)
        pixels_values = img.reshape((-1, 3))
        pixels_values /= 255.0

        cov_matrix = np.cov(pixels_values, rowvar=False)
        eigen_vals, pca_basis = np.linalg.eigh(cov_matrix)

        pixels_project = np.dot(pixels_values, pca_basis)

        new_basis = pca_basis + perturb
        new_basis /= np.sqrt(np.sum(new_basis ** 2, axis=0))[np.newaxis,:]

        pixels_rgb = np.dot(pixels_project, new_basis.T) * 255
        new_img = pixels_rgb.reshape(img.shape)
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)

        return new_img, mask, angles







