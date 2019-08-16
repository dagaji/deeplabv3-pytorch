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


def general_form(rho, theta):
	a = math.cos(theta)
	b = math.sin(theta)
	c = -rho
	return (a,b,c)

def normal_form(a,b,c):
	theta = math.atan(b/a)
	rho = -c
	return (rho, theta)

def find_intersect(line1_coeffs, line2_coeffs):

	A1, B1, C1 = line1_coeffs
	A2, B2, C2 = line2_coeffs

	denom = (A1 * B2 - B1 * A2)

	if abs(denom) > 1e-10:
		x = (B1 * C2 - C1 * B2) / denom
		y = (C1 * A2 - A1 * C2) / denom
	else:
		return None

	return (x, y)

def get_line_coeffs(point, orientation):
	x, y = point
	A = math.cos(orientation)
	B = math.sin(orientation)
	C = -(A * x + B * y)
	return (A, B, C)


def find_intesect_borders(line_coeffs, sz, is_vertical=True):

	H, W = sz

	upper_border_coeffs = (0.0, 1.0, 0.0)
	lower_border_coeffs = (0.0, 1.0, -float(H))
	left_border_coeffs = (1.0, 0.0, 0.0)
	right_border_coeffs = (1.0, 0.0, -float(W))

	if is_vertical:
		border_1_coeffs = upper_border_coeffs
		border_2_coeffs = lower_border_coeffs
		idx = 0
	else:
		border_1_coeffs = left_border_coeffs
		border_2_coeffs = right_border_coeffs
		idx = 1

	border_1_intersect = find_intersect(line_coeffs, border_1_coeffs)
	border_2_intersect = find_intersect(line_coeffs, border_2_coeffs)
	intersect_points = None
	if (border_1_intersect is not None) and (border_2_intersect is not None):
		intersect_points = (border_1_intersect[idx], border_2_intersect[idx])
	return intersect_points


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


def vis_grid(img, lines):

	grid = np.zeros(img.shape, dtype=np.uint8)
	for line in lines:

		rho, theta = line
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(grid, (x1,y1), (x2,y2), (0,0,255), 8)

	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	vis_img = vis.vis_seg(np.squeeze(img[...,::-1]), grid2, vis.make_palette(2))

	fig, ax = plt.subplots(1)
	ax.imshow(vis_img)
	return fig, ax


def vis_grid2(img, lines):

	grid = np.zeros(img.shape, dtype=np.uint8)
	for line in lines:
		pt1 = tuple(np.array(line[0]).astype(int).tolist())
		pt2 = tuple(np.array(line[1]).astype(int).tolist())
		cv2.line(grid, pt1, pt2, (0,0,255), 1)

	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	vis_img = vis.vis_seg(np.squeeze(img[...,::-1]), grid2, vis.make_palette(2))

	fig, ax = plt.subplots(1)
	ax.imshow(vis_img)
	return fig, ax


def make_parser():
	p = ArgumentParser()
	p.add_argument('--dataset', type=str, required=True)
	p.add_argument('--dst', type=str, required=True)
	return p


def sort_lines(lines, sz, is_vertical):

	intersect_points_list = []
	valid_lines = []
	for line in lines:
			line_coeffs = general_form(*line)
			intersect_points = find_intesect_borders(line_coeffs, sz, is_vertical=is_vertical)
			if intersect_points is not None:
				intersect_points_list.append(intersect_points[0])
				valid_lines.append(line_coeffs)

	indices = np.argsort(intersect_points_list).tolist()
	sorted_lines = [valid_lines[idx] for idx in indices]

	return sorted_lines

if __name__ == "__main__":

	args = make_parser().parse_args()

	masks_dir = os.path.join(args.dataset, "masks_test")
	if os.path.exists(args.dst):
		shutil.rmtree(args.dst, ignore_errors=True)
	os.makedirs(args.dst)

	kernel = np.ones((3,3), np.uint8)
	iterations = 2
	angle_range_h = (65.0, 115.0)
	angle_range_v = (-25.0, 25.0)

	for glob in Path(masks_dir).glob("*.png"):

		mask_path = str(glob)

		mask = cv2.imread(mask_path)[...,0]
		instance_seg_mask = np.zeros(mask.shape, dtype=np.uint8)

		img_name = os.path.basename(mask_path)
		save_path = os.path.join(args.dst, img_name)
		if "APR" not in img_name:
			cv2.imwrite(save_path, instance_seg_mask)
			continue
		
		mask[mask != 1] = 0
		mask = cv2.erode(mask, kernel, iterations=iterations)

		_, angles_h, dists_h = search_lines(mask, angle_range_h, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		lines_h = get_lines(dists_h, angles_h)
		sorted_lines_h = sort_lines(lines_h, mask.shape, False)

		_, angles_v, dists_v = search_lines(mask, angle_range_v, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		lines_v = get_lines(dists_v, angles_v)
		sorted_lines_v = sort_lines(lines_v, mask.shape, True)

		
		num_instances = 0
		for idx_v in range(len(sorted_lines_v)-1):
			line_v_1 = sorted_lines_v[idx_v]
			line_v_2 = sorted_lines_v[idx_v+1]
			for idx_h in range(len(sorted_lines_h)-1):

				line_h_1 = sorted_lines_h[idx_h]
				line_h_2 = sorted_lines_h[idx_h+1]

				upper_point1 = find_intersect(line_h_1, line_v_1)
				upper_point2 = find_intersect(line_h_1, line_v_2)

				lower_point1 = find_intersect(line_h_2, line_v_1)
				lower_point2 = find_intersect(line_h_2, line_v_2)

				canvas = PIL.Image.fromarray(np.zeros(mask.shape, dtype=np.uint8))
				draw = PIL.ImageDraw.Draw(canvas)
				num_instances += 1
				draw.polygon(xy=[upper_point1, upper_point2, lower_point2, lower_point1, upper_point1], fill=num_instances)
				canvas = cv2.erode(np.array(canvas), kernel, iterations=12)
				instance_seg_mask[canvas == num_instances] = num_instances

		cv2.imwrite(save_path, instance_seg_mask)











