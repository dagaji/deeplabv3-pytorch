import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from scipy.signal import find_peaks
import deeplabv3.vis as vis
import math
from scipy.spatial import distance

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

	if len(intersect_points) == 2:
		return intersect_points
	else:
		return None


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
		pt1 = tuple(np.array(line[0]).astype(int).tolist())
		pt2 = tuple(np.array(line[1]).astype(int).tolist())
		cv2.line(grid, pt1, pt2, (0,0,255), 1)

	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	vis_img = vis.vis_seg(np.squeeze(img[...,::-1]), grid2, vis.make_palette(2))

	return vis_img


def get_anchors(orientation, M, sz, is_vertical):

	orientation_rad = np.deg2rad(orientation)
	center_point = tuple(((np.array(sz[::-1]) - 1) / 2).tolist())
	line_coeffs = get_line_coeffs(center_point, np.pi/2 - orientation_rad)
	intersect_points = find_intesect_borders(line_coeffs, sz, is_vertical=(not is_vertical))
	step_len = distance.euclidean(intersect_points[0], intersect_points[1]) / M
	unit_vector = np.array((np.cos(orientation_rad), np.sin(orientation_rad)))
	anchor_lines = []

	for i in range(1, M):
		anchor_point = np.array(intersect_points[0]) + i * step_len * unit_vector
		line_coeffs_anchor = get_line_coeffs(tuple(anchor_point.tolist()), orientation_rad)
		anchor_lines.append(normal_form(*line_coeffs_anchor))

	return get_intersect_points(anchor_lines, sz, is_vertical=is_vertical)



def get_intersect_points(lines, sz, is_vertical):

	intersect_points_list = []
	for line in lines:
			line_coeffs = general_form(*line)
			intersect_points = find_intesect_borders(line_coeffs, sz, is_vertical=is_vertical)
			if intersect_points is not None:
				intersect_points_list.append(intersect_points)

	return intersect_points_list