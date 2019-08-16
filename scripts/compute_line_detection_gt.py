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

def check_intersect(point, sz):

	H, W = sz

	if point:
		exists_intersect = (0.0 <= point[0] <= float(W)) and (0.0 <= point[1] <= float(H))
	else:
		exists_intersect = False

	return exists_intersect

def find_intesect_borders(line_coeffs, sz, is_vertical=True):

	H, W = sz

	upper_border_coeffs = (0.0, 1.0, 0.0)
	lower_border_coeffs = (0.0, 1.0, -float(H))
	left_border_coeffs = (1.0, 0.0, 0.0)
	right_border_coeffs = (1.0, 0.0, -float(W))

	upper_border_intersect = find_intersect(line_coeffs, upper_border_coeffs)
	lower_border_intersect = find_intersect(line_coeffs, lower_border_coeffs)
	left_border_intersect = find_intersect(line_coeffs, left_border_coeffs)
	right_border_intersect = find_intersect(line_coeffs, right_border_coeffs)

	intersect_points = []
	if is_vertical:
		if check_intersect(upper_border_intersect, sz):
			intersect_points.append(upper_border_intersect)
		if check_intersect(lower_border_intersect, sz):
			intersect_points.append(lower_border_intersect)
	else:
		if check_intersect(left_border_intersect, sz):
			intersect_points.append(left_border_intersect)
		if check_intersect(right_border_intersect, sz):
			intersect_points.append(right_border_intersect)

	#pdb.set_trace()

	if len(intersect_points) == 2:
		return intersect_points
	else:
		return None



def rotate_img(mat, angle, bbox_size, is_mask=False):

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
		rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_NEAREST, borderValue=255)
	else:
		rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_AREA, borderValue=0)

	rotated_center = (int(bound_w/2), int(bound_h/2))

	x_limits = np.array((-bbox_size[0]/2, bbox_size[0]/2 + bbox_size[0]%2 - 1)).astype(int) + rotated_center[0]
	y_limits = np.array((-bbox_size[1]/2, bbox_size[1]/2 + bbox_size[1]%2 - 1)).astype(int) + rotated_center[1]

	bbox = rotated_mat[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]

	return bbox

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

def get_anchors(orientation, M, sz, is_vertical, plot=False):

	orientation_rad = np.deg2rad(orientation)
	center_point = tuple(((np.array(sz[::-1]) - 1) / 2).tolist())
	line_coeffs = get_line_coeffs(center_point, np.pi/2 - orientation_rad)
	intersect_points = find_intesect_borders(line_coeffs, sz, is_vertical=(not is_vertical))
	step_len = distance.euclidean(intersect_points[0], intersect_points[1]) / M
	unit_vector = np.array((np.cos(orientation_rad), np.sin(orientation_rad)))
	anchor_lines = []
	if plot:
		fig, ax = plt.subplots(1)
		ax.imshow(np.zeros(sz + (3,), dtype=np.uint8))
	for i in range(1, M):
		anchor_point = np.array(intersect_points[0]) + i * step_len * unit_vector
		line_coeffs_anchor = get_line_coeffs(tuple(anchor_point.tolist()), orientation_rad)
		anchor_lines.append(normal_form(*line_coeffs_anchor))
		if plot:
			circle = plt.Circle(tuple(anchor_point.tolist()), 2, color='b')
			ax.add_patch(circle)
	if plot:
		plt.show()

	return get_intersect_points(anchor_lines, sz, is_vertical=is_vertical)

def get_intersect_points(lines, sz, is_vertical):

	intersect_points_list = []
	for line in lines:
			line_coeffs = general_form(*line)
			intersect_points = find_intesect_borders(line_coeffs, sz, is_vertical=is_vertical)
			if intersect_points is not None:
				intersect_points_list.append(intersect_points)

	return intersect_points_list

# def compute_gt(anchor_lines, lines, dist_tresh=7.5, plot=False):

# 	clf_gt = []
# 	regression_gt = []
# 	for anchor_line in anchor_lines:
# 		anchor_offsets = []
# 		anchor_dist = []
# 		anchor_line_coords = np.array(anchor_line).reshape(-1,4).squeeze()
# 		for line in lines:
# 			line_coords = np.array(line).reshape(-1,4).squeeze()
# 			offsets = anchor_line_coords - line_coords
# 			dist = np.abs(offsets[np.abs(offsets) > 0]).sum() / 2.0
# 			anchor_offsets.append(offsets)
# 			anchor_dist.append(dist)
# 		min_dist_index = np.argmin(anchor_dist)
# 		min_dist = anchor_dist[min_dist_index]
# 		if min_dist <= dist_tresh:
# 			clf_prob = 1.0
# 			regression_offset = anchor_offsets[min_dist_index].tolist()
# 			if plot:
# 				vis_grid2(np.zeros((449, 833, 3), dtype=np.uint8), [lines[min_dist_index], anchor_line])
# 				plt.show()
# 		else:
# 			clf_prob = 0.0
# 			regression_offset = [0, 0, 0, 0]
# 		clf_gt.append(clf_prob)
# 		regression_gt.append(regression_offset)
# 	return clf_gt, regression_gt

def compute_gt(anchor_lines, lines, dist_tresh=20):

	clf_gt = np.zeros(len(anchor_lines), dtype=np.float32)
	for idx, anchor_line in enumerate(anchor_lines):
		anchor_offsets = []
		anchor_dist = []
		anchor_line_coords = np.array(anchor_line).reshape(-1,4).squeeze()
		for line in lines:
			line_coords = np.array(line).reshape(-1,4).squeeze()
			offsets = anchor_line_coords - line_coords
			dist = np.abs(offsets[np.abs(offsets) > 0]).sum() / 2.0
			anchor_offsets.append(offsets)
			anchor_dist.append(dist)
		min_dist_index = np.argmin(anchor_dist)
		min_dist = anchor_dist[min_dist_index]
		clf_gt[idx] = np.float32(min_dist <= dist_tresh)

	return clf_gt




if __name__ == "__main__":

	args = make_parser().parse_args()

	images_dir = os.path.join(args.dataset, "images")
	masks_dir = os.path.join(args.dataset, "masks_test")

	dst_images_dir = os.path.join(args.dst, "images")
	if os.path.exists(args.dst):
		shutil.rmtree(args.dst, ignore_errors=True)
	os.makedirs(dst_images_dir)

	kernel = np.ones((3,3), np.uint8)
	iterations = 2
	bbox_size = (833, 449)
	angle_range_coarse = (45.0, 135.0)
	angle_range_h = (95.0, 85.0)
	angle_range_v = (-5.0, 5.0)

	anchors_v = get_anchors(0.0, 50, bbox_size[::-1], is_vertical=True, plot=False)
	anchors_h = get_anchors(90.0, 25, bbox_size[::-1], is_vertical=False, plot=False)

	clf_gt = dict()
	for glob in Path(masks_dir).glob("*.png"):

		mask_path = str(glob)
		img_name = os.path.basename(mask_path)
		img_path = os.path.join(images_dir, img_name)
		dst_img_path = os.path.join(dst_images_dir, img_name)
		if "APR" not in img_name:
			continue
		img = cv2.imread(img_path)
		mask = cv2.imread(mask_path)[...,0]
		mask[mask != 1] = 0
		mask = cv2.erode(mask, kernel, iterations=iterations)
		_, angles, _ = search_lines(mask, angle_range_coarse, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		rotation_angle = 90.0 - np.rad2deg(angles[0])
		img_rotated = rotate_img(img, -rotation_angle, bbox_size)
		cv2.imwrite(dst_img_path, img_rotated)
		mask_rotated = rotate_img(mask, -rotation_angle, bbox_size, is_mask=True)

		_, angles_h, dists_h = search_lines(mask_rotated, angle_range_h, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		lines_h = get_lines(dists_h, angles_h)
		intersect_points_h = get_intersect_points(lines_h, bbox_size[::-1], is_vertical=False)

		_, angles_v, dists_v = search_lines(mask_rotated, angle_range_v, npoints=1000, min_distance=100, min_angle=300, threshold=None)
		lines_v = get_lines(dists_v, angles_v)
		intersect_points_v = get_intersect_points(lines_v, bbox_size[::-1], is_vertical=True)
		
		clf_gt_h = compute_gt(anchors_h, intersect_points_h)
		clf_gt_v = compute_gt(anchors_v, intersect_points_v)
		clf_gt[img_name] = dict()
		clf_gt[img_name]['h'] = clf_gt_h
		clf_gt[img_name]['v'] = clf_gt_v

	with open(os.path.join(args.dst, 'lines_gt.pickle'),'wb') as file:
		pickle.dump(clf_gt, file)



		














