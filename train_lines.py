import torch
from deeplabv3.metrics import AverageMeter
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import deeplabv3.dataset.lines as lines_datasets
import deeplabv3.model.lines as lines_models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
from argparse import ArgumentParser
import deeplabv3.lines as lines
from torch.nn import functional as F
import cv2
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

bbox_size = (833, 449)

anchors_v_intersect = lines.get_anchors(0.0, 50, bbox_size[::-1], is_vertical=True)
anchors_h_intersect = lines.get_anchors(90.0, 25, bbox_size[::-1], is_vertical=False)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--dataset', type=str, default='APR_TAX_RWY')
	parser.add_argument('--partition', type=int, default=3)
	return parser.parse_args()

def train(features, line_loader, iter_size):

	for inner_iter, (grids, gts) in enumerate(line_loader):
		grids = grids.to(device)
		gts = gts.to(device)
		weight = (1.0 if gts.cpu().numpy()[0] > 0.5 else 1.0) * torch.ones(1).float().to(device)
		pos_weight = torch.ones([64])
		lines_scores = detection_model(features, grids)
		loss = F.binary_cross_entropy_with_logits(lines_scores, torch.unsqueeze(gts,dim=1), pos_weight=weight) / iter_size
		#loss = criterion(lines_scores, torch.unsqueeze(gts,dim=1), pos_weight=weight) / iter_size
		loss.backward()
		running_loss.update(loss.item() * iter_size, n=grids.size(0))

def validate(features, image_id):

	def val_dir(line_loader):
		line_probs = np.zeros(len(line_loader))
		for idx, (grids, _) in enumerate(line_loader):
			grids = grids.to(device)
			lines_scores = detection_model(features, grids)
			line_probs[idx] = torch.sigmoid(torch.squeeze(lines_scores)).cpu().numpy()
		return line_probs

	line_dataset_v, line_dataset_h = lines_factory.get_line_datasets(image_id[0])
	line_loader_v = torch.utils.data.DataLoader(line_dataset_v, batch_size=1, shuffle=True, num_workers=0)
	line_loader_h = torch.utils.data.DataLoader(line_dataset_h, batch_size=1, shuffle=True, num_workers=0)

	line_probs_v = val_dir(line_loader_v)
	line_probs_h = val_dir(line_loader_h)

	detected_lines_v = np.array(anchors_v_intersect)[line_probs_v > 0.5].tolist()
	detected_lines_h = np.array(anchors_h_intersect)[line_probs_h > 0.5].tolist()

	return detected_lines_v + detected_lines_h



root = '/home/davidgj/projects/APR_rotated'
checkpoint_dir = '/home/davidgj/projects/deeplabv3-pytorch/checkpoint'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parse_args()

checkpoint_path = os.path.join(checkpoint_dir, 
							   args.dataset, 
							   'partition_{}', 
							   'decoder-s8-res1000-multitask', 
							   'epoch_8.pth').format(args.partition)
seg_model = lines_models.get_seg_model().to(device)
checkpoint_dict = torch.load(checkpoint_path)
seg_model.load_state_dict(checkpoint_dict["model_state_dict"], strict=False)
for param in seg_model.parameters():
	param.requires_grad = False
seg_model.eval()
detection_model = lines_models.LineDetectionNet().to(device)
detection_model.train()

list_dir = os.path.join('list', args.dataset, 'partition_{}').format(args.partition)
train_id_txt = os.path.join(list_dir, 'train.txt')
val_id_txt = os.path.join(list_dir, 'val.txt')

img_dataset_train = lines_datasets.RotatedImagesDataset(root, train_id_txt)
img_dataset_val = lines_datasets.RotatedImagesDataset(root, val_id_txt)

img_loader_train = torch.utils.data.DataLoader(img_dataset_train, batch_size=1, shuffle=True, num_workers=0)
img_loader_val = torch.utils.data.DataLoader(img_dataset_val, batch_size=1, shuffle=True, num_workers=0)

lines_factory = lines_datasets.LinesDatasetFactory(root)

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

params_to_update = detection_model.parameters()
optimizer = optim.SGD(params_to_update, lr= 0.0005, momentum=0.9, weight_decay=1e-5)

running_loss = AverageMeter()
running_loss.reset()
for _ in range(6):
	for outer_iter, data in tqdm(enumerate(img_loader_train), total=len(img_loader_train), dynamic_ncols=True):

		image_id = data['image_id']
		image = data['image'].to(device)
		features = seg_model(image)
		line_dataset_v, line_dataset_h = lines_factory.get_line_datasets(image_id[0])
		line_loader_v = torch.utils.data.DataLoader(line_dataset_v, batch_size=1, shuffle=True, num_workers=0)
		line_loader_h = torch.utils.data.DataLoader(line_dataset_h, batch_size=1, shuffle=True, num_workers=0)

		optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			train(features, line_loader_v, len(line_loader_v) + len(line_loader_h))
			train(features, line_loader_h, len(line_loader_v) + len(line_loader_h))
		optimizer.step()

		if outer_iter % 10 == 0:
			print('>>> Loss: {:.4f}'.format(running_loss.value()))
			running_loss.reset()
			
	detection_model.eval()
	with torch.set_grad_enabled(False):
		val_iterator = iter(img_loader_train)
		val_data = next(val_iterator)
		val_image = val_data['image'].to(device)
		val_id = val_data['image_id']
		val_features = seg_model(val_image)
		detected_lines = validate(val_features, val_id)
	detection_model.train()

	_val_img = cv2.imread(os.path.join(root,'images', val_id[0]))
	vis_img = lines.vis_grid(_val_img, detected_lines)
	plt.figure()
	plt.imshow(vis_img)

plt.show()
	







