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

    def __call__(self, img, mask):
        for aug in self.augmentations:
            img, mask = aug(img, mask)
        return img, mask


@register.attach('random_crop')
class _RandomCrop(object):

    def __init__(self, w, h):
        self.bbox_size = (w, h)

    def crop(self, img, top_border):
        x_limits = (top_border[0], top_border[0] + self.bbox_size[0])
        y_limits = (top_border[1], top_border[1] + self.bbox_size[1])
        bbox = img[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]
        return bbox

    def __call__(self, img, mask):
        h, w = img.shape[:2]
        x_max = w - self.bbox_size[0]
        y_max = h - self.bbox_size[1]
        top_border_x = np.random.uniform(0, x_max, 1).astype(int)
        top_border_y = np.random.uniform(0, y_max, 1).astype(int)
        top_border = (top_border_x[0], top_border_y[0])
        cropped_img = self.crop(img, top_border)
        cropped_mask = self.crop(mask, top_border)
        return cropped_img, cropped_mask


@register.attach('random_crop_mosaic')
class _RandomCropMosaic(object):

    def __init__(self, bbox_scale=0.75, max_offset=0.1, resize_scale=None):
        self.bbox_scale = bbox_scale
        self.max_offset = max_offset
        self.resize_scale = resize_scale

    def crop(self, img, top_border, bbox_size):
        x_limits = (top_border[0], top_border[0] + bbox_size[0])
        y_limits = (top_border[1], top_border[1] + bbox_size[1])
        bbox = img[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]
        return bbox

    def __call__(self, img, mask):

        h, w = img.shape[:2]
        bbox_size = (int(self.bbox_scale * w), int(self.bbox_scale * h))
        offset = np.random.uniform(-1.0 * self.max_offset, self.max_offset, 1)
        # print("offset x: {}".format(offset))
        top_border_x = w / 2 + offset * w - bbox_size[0] / 2
        offset = np.random.uniform(-1.0 * self.max_offset, self.max_offset, 1)
        # print("offset y: {}".format(offset))
        top_border_y = h / 2 + offset * h - bbox_size[1] / 2
        top_border = (top_border_x[0].astype(int), top_border_y[0].astype(int))
        cropped_img = self.crop(img, top_border, bbox_size)
        cropped_mask = self.crop(mask, top_border, bbox_size)
        if self.resize_scale is not None:
            new_bbox_size = (int(bbox_size[0] * self.resize_scale),
                             int(bbox_size[1] * self.resize_scale))
            cropped_img = cv2.resize(cropped_img, new_bbox_size)
            cropped_mask = cv2.resize(cropped_mask, new_bbox_size, interpolation=cv2.INTER_NEAREST)
        # pdb.set_trace()
        # plt.figure()
        # plt.imshow(cropped_img)
        # plt.figure()
        # plt.imshow(cropped_mask)
        # plt.show()
        return cropped_img, cropped_mask


@register.attach('random_flip')
class _RandomFlip(object):

    def __call__(self, img, mask):
        flip_op = np.random.choice([-1,0,1,2])
        if flip_op != 2:
            img = cv2.flip(img, flip_op)
            mask = cv2.flip(mask, flip_op)
        return img, mask
        

@register.attach('rotate')
class _RandomRotation(object):

    def __init__(self, sigma_angle, w, h, aux_index=100, ignore_index=255):
        self.angle_gen = _get_random_gen(sigma_angle)
        self.crop_size = (w, h)
        self.aux_index = aux_index
        self.ignore_index = ignore_index

    def __call__(self, img, label):

        angle = self.angle_gen.rvs()
        img = self.rotate_img(img, angle, self.crop_size)
        label = self.rotate_img(label, angle, self.crop_size, is_mask=True)

        ignore_mask = label.copy()
        ignore_mask[ignore_mask != self.aux_index] = 0
        kernel = np.ones((21,21), np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel, iterations=3)
        label[ignore_mask == self.aux_index] = self.ignore_index

        return img, label

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


@register.attach('choice')
class _RandomChoice(object):

    def __init__(self, augmentations_list):
        self.augmentations = []
        for el in augmentations_list:
            aug_name = str(*el.keys())
            params = el[aug_name]
            self.augmentations.append(_get_augmentation(aug_name, params))

    def __call__(self, img, mask):
        if len(self.augmentations):
            indices = range(len(self.augmentations))
            idx = np.random.choice(indices)
            img, mask = self.augmentations[idx](img, mask)
        return img, mask


@register.attach('pca')
class _PCAColorAugmentation(object):

    def __init__(self, sigma_pca):
        self.perturb_gen = _get_random_gen(sigma_pca, alfa=1.0)

    def __call__(self, img, mask):

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

        return new_img, mask


@register.attach('pca-pano')
class _PCAColorAugmentationPano(object):

    def __init__(self, sigma_pca):
        self.perturb_gen = _get_random_gen(sigma_pca, alfa=1.0)

    def __call__(self, img, mask):

        # plt.figure()
        # plt.imshow(img)

        perturb = self.perturb_gen.rvs((3,3))
        pixels_mask = np.logical_and(img[...,0] == 0, img[...,1] == 0)
        pixels_mask = np.logical_and(pixels_mask, img[...,2] == 0)

        img = img.astype(float)

        pixels_values = img[pixels_mask]
        pixels_values /= 255.0

        cov_matrix = np.cov(pixels_values, rowvar=False)
        eigen_vals, pca_basis = np.linalg.eigh(cov_matrix)

        pixels_values = img.reshape((-1, 3))
        pixels_values /= 255.0
        pixels_project = np.dot(pixels_values, pca_basis)

        new_basis = pca_basis + perturb
        new_basis /= np.sqrt(np.sum(new_basis ** 2, axis=0))[np.newaxis,:]

        pixels_rgb = np.dot(pixels_project, new_basis.T) * 255
        new_img = pixels_rgb.reshape(img.shape)
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)

        # plt.figure()
        # plt.imshow(new_img)
        # plt.show()

        return new_img, mask







