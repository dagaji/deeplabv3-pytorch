import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from scipy.signal import find_peaks
import deeplabv3.vis as vis
import math
from scipy.spatial import distance

class LineSampler:

	def __init__(self, angle_step=1.0, rho_step=10, npoints=50, plot=False):

		self.angle_step = angle_step
		self.rho_step = rho_step
		self.npoints = npoints
		self.plot = plot


	def __call__(self, angle_range, sz):

		def norm_coords(coords):
			coords[:, 0] = 2 * coords[:,0] / float(sz[1] - 1) - 1
			coords[:, 1] = 2 * coords[:,1] / float(sz[0] - 1) - 1
			return coords

		def line_coords(intersect_points, orientation):

			if self.plot:
				fig, ax = plt.subplots(1)
				ax.imshow(np.zeros(sz + (3,), dtype=np.uint8))
				circle = plt.Circle(intersect_points[0], 5, color='b')
				ax.add_patch(circle)
				circle = plt.Circle(intersect_points[1], 5, color='b')
				ax.add_patch(circle)

			if self.plot:
				fig, ax = plt.subplots(1)
				ax.imshow(np.zeros(sz + (3,), dtype=np.uint8))

			step_len = distance.euclidean(intersect_points[0], intersect_points[1]) / self.npoints
			diff_vector = np.array(intersect_points[1]) - np.array(intersect_points[0])
			unit_vector = diff_vector / np.sqrt((diff_vector ** 2).sum())
			line_points = []
			for i in np.arange(1, self.npoints):
				line_point = np.array(intersect_points[0]) + i * step_len * unit_vector
				line_points.append(line_point)
				if self.plot:
					circle = plt.Circle(tuple(line_point.tolist()), 2, color='b')
					ax.add_patch(circle)

			if self.plot:
				plt.show()

			return np.array(line_points).astype(np.float32)

		max_distance = 2 * np.sqrt(sz[0] ** 2 + sz[1] ** 2)
		rhos =  np.arange(-max_distance / 2.0, max_distance / 2.0 + self.rho_step, self.rho_step)
		thetas = np.arange(angle_range[0], angle_range[1] + self.angle_step, self.angle_step)
		thetas = np.deg2rad(thetas)

		sampled_lines = []
		lines_coeffs = []
		for _theta in thetas.tolist():
			for _rho in rhos.tolist():
				line_coeffs = general_form(_rho, _theta)
				intersect_points = find_intesect_borders(line_coeffs, sz)
				if intersect_points is not None:
					line_points = line_coords(intersect_points, _theta)
					line_points = norm_coords(line_points)
					sampled_lines.append(line_points)
					lines_coeffs.append(normal_form(*line_coeffs))

		return sampled_lines, lines_coeffs


def points2line_eq(point1, point2):

	x1, y1 = point1
	x2, y2 = point2

	a = float(y2 - y1)
	b = float(x1 - x2)
	c = -(a * x1 + b * y1)
	scale_factor = 1.0 / np.sqrt(a ** 2 + b ** 2)
	return tuple((np.array((a, b, c)) * scale_factor).tolist())

def general_form(rho, theta):
	if rho < 0.0:
		theta += np.pi
		rho *= -1.0
	a = math.cos(theta)
	b = math.sin(theta)
	c = -rho
	return (a,b,c)

def normal_form(a,b,c):

	theta = np.arctan2(b,a)
	scale_factor = 1.0 / np.sqrt(a ** 2 + b ** 2)
	rho = -c * scale_factor
	if rho < 0:
		rho *= -1.0
		theta += np.pi
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
		exists_intersect = (-1.0 <= point[0] <= float(W+1)) and (-1.0 <= point[1] <= float(H+1))
	else:
		exists_intersect = False

	return exists_intersect

def find_intesect_borders(line_coeffs, sz):

	def remove_intersect(intersect_points, anchor_idx):
		intersect_points = np.array(intersect_points)
		anchor_point = intersect_points[anchor_idx]
		dist = np.sqrt(np.sum((anchor_point - intersect_points) ** 2, axis=1))
		dist[anchor_idx] = np.inf
		return intersect_points[dist > 100].tolist()

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
	if check_intersect(upper_border_intersect, sz):
		intersect_points.append(upper_border_intersect)
	if check_intersect(lower_border_intersect, sz):
		intersect_points.append(lower_border_intersect)
	if check_intersect(left_border_intersect, sz):
		intersect_points.append(left_border_intersect)
	if check_intersect(right_border_intersect, sz):
		intersect_points.append(right_border_intersect)

	if len(intersect_points) == 2:
		return intersect_points
	elif len(intersect_points) > 2:
		intersect_points = remove_intersect(intersect_points, 0)
		intersect_points = remove_intersect(intersect_points, 1)
		return [tuple(intersect_points[0]), tuple(intersect_points[1])]
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


def create_grid_intersect(lines, sz):

	grid = np.zeros(sz + (3,), dtype=np.uint8)
	for line in lines:
		pt1 = tuple(np.array(line[0]).astype(int).tolist())
		pt2 = tuple(np.array(line[1]).astype(int).tolist())
		cv2.line(grid, pt1, pt2, (0,0,255), 1)

	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	return grid2



def vis_grid(img, lines):

	grid = np.zeros(img.shape, dtype=np.uint8)
	for line in lines:
		pt1 = tuple(np.array(line[0]).astype(int).tolist())
		pt2 = tuple(np.array(line[1]).astype(int).tolist())
		cv2.line(grid, pt1, pt2, (0,0,255), 1)

	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	vis_img = vis.vis_seg(np.squeeze(img[...,::-1]), grid2, vis.make_palette(2))

	return vis_img

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
		cv2.line(grid, (x1,y1), (x2,y2), (0,0,255), width)
	grid2 = (grid[...,-1] == 255).astype(np.uint8)
	return grid2

def compute_hist(hspace, angles, bin_length=10):

	M, N = hspace.shape
	hist = np.zeros(int(N/bin_length), dtype=float)
	j = 0
	bins_edges = []
	for i in range(0, N, bin_length):
		hist[j] = np.sum(hspace[:,i:i+bin_length])
		bins_edges.append(angles[i])
		j += 1
	return hist, np.rad2deg(bins_edges)


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