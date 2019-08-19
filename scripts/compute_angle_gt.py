import matplotlib
matplotlib.use('tkagg')
from pathlib import Path
import os
import os.path
import shutil
from argparse import ArgumentParser
import pdb
import cv2
from distutils.dir_util import copy_tree
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from scipy.signal import find_peaks
import vis
from matplotlib.patches import Circle
import math
from scipy.spatial import distance
import pickle
import PIL.Image
import PIL.ImageDraw
from shutil import copyfile
import pickle


def general_form(rho, theta):
	a = math.cos(theta)
	b = math.sin(theta)
	c = -rho
	return (a,b,c)

def normal_form(a,b,c):
	theta = math.atan(b/a)
	rho = -c
	return (rho, theta)

def get_line_coeffs(point, orientation):
	x, y = point
	A = math.cos(orientation)
	B = math.sin(orientation)
	C = -(A * x + B * y)
	return (A, B, C)


def get_lines(dist, angles):

	lines = []
	for rho, theta in zip(dist, angles):
		if rho < 0:
			rho *= -1
			theta += np.pi
		lines.append((rho, theta))
	return lines


def search_lines(blob, angle_range, npoints=1000, min_distance=100, min_angle=300, threshold=None):

	thetas = np.linspace(np.deg2rad(angle_range[0]), np.deg2rad(angle_range[1]), npoints)
	hspace, angles, distances = hough_line(blob, thetas)

	if threshold is not None:
		accum, angles, dists = hough_line_peaks(hspace, angles, distances, min_distance=min_distance, min_angle=min_angle, threshold=threshold * np.max(hspace))
	else:
		accum, angles, dists = hough_line_peaks(hspace, angles, distances, min_distance=min_distance, min_angle=min_angle)
	
	return accum, angles, dists


def create_grid(sz, lines, width=8):

	grid = np.zeros(sz+(3,), dtype=np.uint8)
	for line in lines:
		rho, theta = line
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1500*(-b))
		y1 = int(y0 + 1500*(a))
		x2 = int(x0 - 1500*(-b))
		y2 = int(y0 - 1500*(a))
		cv2.line(grid, (x1,y1), (x2,y2), (0,0,255), 8)
	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	return grid2


def make_parser():
	p = ArgumentParser()
	p.add_argument('--dataset', type=str, required=True)
	p.add_argument('--dst', type=str, required=True)
	return p


if __name__ == "__main__":

	args = make_parser().parse_args()

	masks_dir = os.path.join(args.dataset, "masks_test")
	images_dir = os.path.join(args.dataset, "images")

	if os.path.exists(args.dst):
		shutil.rmtree(args.dst, ignore_errors=True)
	dst_images_dir = os.path.join(args.dst, "images")
	dst_masks_test_dir = os.path.join(args.dst, "masks_test")
	dst_masks_dir = os.path.join(args.dst, "masks")
	os.makedirs(dst_images_dir)
	os.makedirs(dst_masks_test_dir)
	os.makedirs(dst_masks_dir)

	kernel = np.ones((3,3), np.uint8)
	iterations = 2
	angle_range_h = (55.0, 125.0)
	angle_range_v = (-35.0, 35.0)

	imgs_angles = dict()
	mean_angle_h_list = []
	mean_angle_v_list = []

	for glob in Path(masks_dir).glob("*.png"):

		mask_path = str(glob)
		img_name = os.path.basename(mask_path)
		img_path = os.path.join(images_dir, img_name)
		
		dst_img_path = os.path.join(dst_images_dir, img_name)
		dst_mask_path = os.path.join(dst_masks_dir, img_name)
		dst_mask_test_path = os.path.join(dst_masks_test_dir, img_name)
		if "APR" not in img_name:
			copyfile(img_path, dst_img_path)
			copyfile(mask_path, dst_mask_path)
			copyfile(mask_path, dst_mask_test_path)
			continue
		
		mask = cv2.imread(mask_path)[...,0]
		mask_1 = mask.copy()
		mask_1[mask_1 != 1] = 0
		mask_1 = cv2.erode(mask_1, kernel, iterations=iterations)

		_, angles_h, dists_h = search_lines(mask_1, angle_range_h, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		lines_h = get_lines(dists_h, angles_h)
		mean_angle_h = np.rad2deg(angles_h).mean()
		mean_angle_h_list.append(mean_angle_h)

		_, angles_v, dists_v = search_lines(mask_1, angle_range_v, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		lines_v = get_lines(dists_v, angles_v)
		mean_angle_v = np.rad2deg(angles_v).mean()
		mean_angle_v_list.append(mean_angle_v)

		grid16 = create_grid(mask.shape, lines_h + lines_v, width=16)
		grid8 = create_grid(mask.shape, lines_h + lines_v, width=8)

		grid_test = grid16.copy()
		grid_test[grid16 == 1] = 255
		grid_test[grid8 == 1] = 1

		APR_area = np.logical_or(mask == 0, mask == 1)

		dst_mask = mask.copy()
		dst_mask[APR_area] = grid16[APR_area]

		dst_mask_test = mask.copy()
		dst_mask_test[APR_area] = grid_test[APR_area]

		copyfile(img_path, dst_img_path)
		cv2.imwrite(dst_mask_path, dst_mask)
		cv2.imwrite(dst_mask_test_path, dst_mask_test)

		imgs_angles[img_name] = (mean_angle_h, mean_angle_v)

	print("min angle_h: {}, max_angle_h: {}".format(np.min(mean_angle_h_list), np.max(mean_angle_h_list)))
	print("min angle_v: {}, max_angle_v: {}".format(np.min(mean_angle_v_list), np.max(mean_angle_v_list)))

	with open(os.path.join(args.dst, 'angles.pickle'), 'wb') as file:
		pickle.dump(imgs_angles, file)










