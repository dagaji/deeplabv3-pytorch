import torch
from deeplabv3.metrics import AverageMeter
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pdb
import torch.nn as nn
from train import set_bn_eval
import deeplabv3.dataset as dataset
import deeplabv3.model as model

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--dataset', type=str, default='APR_TAX_RWY')
	parser.add_argument('--partition', type=int, default=3)
	return parser.parse_args()

root = '/home/davidgj/projects/APR_rotated'
checkpoint_dir = '/home/davidgj/projects/deeplabv3-pytorch/checkpoint'

args = parse_args()
checkpoint_path = os.path.join(checkpoint_dir, 
							   args.dataset, 
							   'partition_{}', 
							   'decoder-s8-res1000-multitask', 
							   'epoch_8.pth').format(args.partition)

checkpoint_dict = torch.load(checkpoint_path)
model_train.load_state_dict(last_checkpoint["model_state_dict"])
			current_epoch = last_checkpoint["epoch"]
			optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])

list_dir = os.path.join('list', args.dataset, 'partition_{}').format(args.partition)
train_id_txt = os.path.join(list_dir, 'train.txt')
val_id.txt = os.path.join(list_dir, 'val.txt')

img_dataset_train = dataset.RotatedImagesDataset(root, train_id_txt)
img_dataset_val = dataset.RotatedImagesDataset(root, val_id_txt)

img_loader_train = torch.utils.data.DataLoader(img_dataset_train, batch_size=1, shuffle=True, num_workers=0)
img_loader_val = torch.utils.data.DataLoader(img_dataset_val, batch_size=1, shuffle=False, num_workers=0)

lines_factory = dataset.LinesDatasetFactory(root)

