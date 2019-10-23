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

    def __call__(self, img, *args):
        for aug in self.augmentations:
            img, args = aug(img, *args)
        return img, args


@register.attach('random_crop_v2')
class _RandomCrop(object):

    def __init__(self, w, h):
        self.bbox_size = (w, h)

    def crop(self, img, top_border):
        x_limits = (top_border[0], top_border[0] + self.bbox_size[0])
        y_limits = (top_border[1], top_border[1] + self.bbox_size[1])
        bbox = img[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]
        return bbox

    def __call__(self, img, *args):
        h, w = img.shape[:2]
        x_max = w - self.bbox_size[0]
        y_max = h - self.bbox_size[1]
        top_border_x = np.random.uniform(0, x_max, 1).astype(int)
        top_border_y = np.random.uniform(0, y_max, 1).astype(int)
        top_border = (top_border_x[0], top_border_y[0])
        cropped_img = self.crop(img, top_border)
        cropped_masks = []
        for mask in args:
            cropped_masks.append(self.crop(mask, top_border))
        return cropped_img, cropped_masks