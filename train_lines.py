import torch
from deeplabv3.metrics import AverageMeter
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from train import set_bn_eval
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

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--dataset', type=str, default='APR_TAX_RWY')
	parser.add_argument('--partition', type=int, default=3)
	return parser.parse_args()

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

list_dir = os.path.join('list', args.dataset, 'partition_{}').format(args.partition)
train_id_txt = os.path.join(list_dir, 'train.txt')
val_id_txt = os.path.join(list_dir, 'val.txt')

img_dataset_train = lines_datasets.RotatedImagesDataset(root, train_id_txt)
img_dataset_val = lines_datasets.RotatedImagesDataset(root, val_id_txt)

img_loader_train = torch.utils.data.DataLoader(img_dataset_train, batch_size=1, shuffle=True, num_workers=0)
img_loader_val = torch.utils.data.DataLoader(img_dataset_val, batch_size=1, shuffle=False, num_workers=0)

lines_factory = lines_datasets.LinesDatasetFactory(root)

criterion = nn.BCELoss()
criterion = criterion.to(device)

params_to_update = detection_model.parameters()
optimizer = optim.SGD(params_to_update, lr= 0.0005, momentum=0.9, weight_decay=1e-5)

running_loss = AverageMeter()
running_loss.reset()

for outer_iter, data in tqdm(enumerate(img_loader_train), total=len(img_loader_train), dynamic_ncols=True):

	image_id = data['image_id']
	image = data['image'].to(device)
	features = seg_model(image)
	line_loader = torch.utils.data.DataLoader(lines_factory.get_line_dataset(image_id[0]), batch_size=15, shuffle=True, num_workers=0)

	optimizer.zero_grad()
	running_loss.reset()
	pdb.set_trace()
	for inner_iter, (grids, gts) in enumerate(line_loader):
		grids = grids.to(device)
		gts = gst.to(device)
		lines_scores = detection_model(features, grids)
		loss = criterion(lines_scores, gts) / len(line_loader)
		loss.backward()
		running_loss.update(loss.item() * len(line_loader), n=grids.size(0))
	optimizer.step()
	print('>>> Loss: {:.4f}'.format(running_loss.value()))





