import torch
from deeplabv3.metrics import RunningScore
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from deeplabv3.dataset import get_dataset
from deeplabv3.model import get_model
from deeplabv3.optimizer import get_optimizer
from deeplabv3.scheduler import get_scheduler
from deeplabv3.loss import get_loss
import deeplabv3.utils as utils
from deeplabv3.metrics import RunningScore, AverageMeter
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
import os
from argparse import ArgumentParser
from torchsummary import summary
from deeplabv3.save import ResultsSaverFactory, CheckpointSaver, VideoSaver
from pathlib import Path
import time

def get_last_checkpoint(checkpoint_dir):

	if not os.path.exists(checkpoint_dir):
		return None
	else:
		checkpoints_globs = list(Path(checkpoint_dir).glob('*.pth'))
		if len(checkpoints_globs) == 0:
			return None
		key = lambda x: int(os.path.basename(str(x)).split('.')[0].split('_')[-1])
		last_checkpoint_path = str(sorted(checkpoints_globs, key=key, )[-1])
		return last_checkpoint_path

# def get_dataloader(id_list_path, dataset_cfg, batch_size, shuffle=True):

# 	dataset_params = dict(dataset_cfg['params'])
# 	dataset_params.update(id_list_path=id_list_path)
# 	dataset = get_dataset(dataset_cfg['name'])(**dataset_params)
# 	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def get_dataloader(dataset_cfg):
	dataset_params = dict(dataset_cfg)
	dataset = get_dataset('video_dataset')(**dataset_params)
	return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--videos', nargs='+', type=str)
	parser.add_argument('--expers', nargs='+', type=str)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


def validate(val_model, val_loader, num_classes, device, saver=None):

	val_model.eval()   # Set model to evaluate mode

	with torch.set_grad_enabled(False):

		# Iterate over data.
		for _iter, (image, frame) in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):

			inputs = image.to(device)
			preds = val_model(inputs)

			if saver is not None:
				saver.save_frame(frame.numpy().astype(np.uint8), preds.cpu().numpy())

	if saver is not None:
		saver.save_video()

if __name__ == "__main__":

	args = parse_args()

	exper_name = os.path.basename(args.config).split(".")[0]

	num_classes, video_cfg = utils.get_cfgs_video(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	_save_dir = os.path.join('videos', exper_name, '{}')
	
	for video_info in video_cfg["videos-info"]:

		if args.videos is not None:
			if video_info["name"] not in args.videos:
				continue

		val_dataloader = get_dataloader(video_info['params'])

		checkpoint_dir = os.path.join('checkpoint', video_info['checkpoint'], exper_name)
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if last_checkpoint_path is not None:
			print("CHECKPOINT")
			last_checkpoint = torch.load(last_checkpoint_path)

		for video_exper in video_cfg["expers"]:

			if args.expers is not None:
				if video_exper["name"] not in args.expers:
					continue

			save_dir = _save_dir.format(video_exper["name"])
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			save_path = os.path.join(save_dir, video_info["name"] + '.MOV')

			model_val = get_model(num_classes, video_exper["model"]).to(device)
			model_val.load_state_dict(last_checkpoint["model_state_dict"], strict=False)

			results_saver = VideoSaver(num_classes, save_path)

			validate(model_val, 
				val_dataloader, 
				num_classes, 
				device, saver=results_saver)





	
	# for glob in Path(os.path.join(dataset_dir)).glob("*"):

	# 	partition_dir = str(glob)
	# 	partition_number = int(glob.parts[-1].split("_")[-1])

	# 	if args.partitions is not None:
	# 		if partition_number not in args.partitions:
	# 			continue
	# 	print('>> Partition {}'.format(partition_number))
	# 	print()

	# 	partition = os.path.join(dataset_dir, 'partition_{}').format(partition_number)
	# 	_id_list_path = os.path.join(partition, '{}.txt')

	# 	model_train = get_model(num_classes, training_cfg["model"], training_cfg['aux_loss']).to(device)

	# 	val_expers = {}
	# 	for _val_exper in val_cfg['val_expers']:
	# 		if "video" in _val_exper['name']:
	# 			model_val = get_model(num_classes, _val_exper["model"]).to(device)
	# 			val_dataloader = get_dataloader(_val_exper['dataset'], shuffle=False)
	# 			val_expers[_val_exper['name']] = dict(model_val=model_val, val_dataloader=val_dataloader, save_path=_val_exper["save_path"])

	# 	checkpoint_dir = os.path.join('checkpoint', args.dataset, 'partition_{}', exper_name).format(partition_number)
	# 	last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
	# 	if last_checkpoint_path is not None:
	# 		print("CHECKPOINT")
	# 		last_checkpoint = torch.load(last_checkpoint_path)
	# 		model_train.load_state_dict(last_checkpoint["model_state_dict"])
	# 		current_epoch = last_checkpoint["epoch"]
	# 	else:
	# 		current_epoch = 0

	# 	for val_exper_name, val_exper in val_expers.items():
	# 		print('>> {}'.format(val_exper_name))
	# 		val_model, val_dataloader = val_exper['model_val'], val_exper['val_dataloader']
	# 		results_saver = VideoSaver(num_classes, val_exper["save_path"])
	# 		val_model.load_state_dict(model_train.state_dict(), strict=False)
	# 		validate(val_model, 
	# 			val_dataloader, 
	# 			num_classes, 
	# 			device, saver=results_saver)


