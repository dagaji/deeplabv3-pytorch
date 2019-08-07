import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from deeplabv3.dataset import get_dataset
from deeplabv3.model import get_model
from deeplabv3.optimizer import get_optimizer
from deeplabv3.scheduler import get_scheduler
from deeplabv3.loss import relax_loss, custom_loss
import deeplabv3.utils as utils
from deeplabv3.metrics import RunningScore, AverageMeter
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
from argparse import ArgumentParser
from torchsummary import summary
from deeplabv3.save import Saver
from pathlib import Path


def train_model(models, dataloaders, criterion, optimizer, scheduler, device, training_cfg, val_cfg, num_classes, num_epochs=1, saver=None):

	running_score = RunningScore(num_classes)
	running_loss = AverageMeter()

	iter_size = training_cfg['iter_size']
	display_iters = training_cfg['display_iters']
	val_epochs = val_cfg['val_epochs']


	for epoch in range(num_epochs):

		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:

			model = models[phase]

			if phase == 'train':
				model.train()  # Set model to training mode
				model.apply(set_bn_eval)
			elif epoch % val_epochs == 0:
				model_train = models['train']
				model.load_state_dict(model_train.state_dict(), strict=True)
				model.eval()   # Set model to evaluate mode
				if saver is not None:
					saver.save_checkpoint(epoch, model_train, optimizer)
			else:
				continue

			running_score.reset()
			running_loss.reset()
			optimizer.zero_grad()

			# Iterate over data.
			for _iter, batch in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), dynamic_ncols=True):

				torch.cuda.empty_cache()

				# image_id = batch.pop('image_id')
				# inputs = batch.pop('image')
				# labels = batch.pop('label')

				# if 'normalizer' in  batch:
				# 	normalizer = batch.pop('normalizer')
				# 	normalizer = normalizer.to(device)

				# inputs = inputs.to(device)
				# labels = labels.to(device)

				image_id = batch.pop('image_id')
				inputs = batch.pop('image').to(device)
				label_1hot = batch.pop('label').to(device)
				mask = batch.pop('mask').to(device)
				refine_mask = batch.pop('refine_mask').to(device)
				
				with torch.set_grad_enabled(phase == 'train'):

					outputs = model(inputs)

					if phase == 'train':

						#loss = criterion(outputs['out'], labels) / iter_size
						loss = criterion(outputs['out'], label_1hot, mask, refine_mask) / iter_size
						loss.backward()
						running_loss.update(loss.item() * iter_size, n=inputs.size(0))
						scheduler.step()

						if _iter > 0 and _iter % iter_size == 0:
							optimizer.step()
							optimizer.zero_grad()

						if _iter > 0 and _iter % (display_iters * iter_size) == 0:
							print('{} Loss: {:.4f}'.format(phase, running_loss.value()))
							for idx, param_group in enumerate(optimizer.param_groups):
								print('Param Group {}: {}'.format(idx, param_group['lr']))
							running_loss.reset()

					elif phase == "val":
						_, preds = torch.max(outputs['out'], 1)
						running_score.update(labels, preds)
						if saver is not None:
							saver.save_vis(image_id, preds.cpu().numpy(), epoch)

			if phase == "val":
				per_class_iu = running_score.get_pre_class_iu()
				print('>>>>>>>> {} Per_class: {}'.format(phase, per_class_iu))
				if saver is not None:
					saver.save_score(per_class_iu, epoch)



def set_bn_eval(m):
	if isinstance(m, nn.modules.batchnorm._BatchNorm):
		m.eval()


def get_dataloaders(partition, training_cfg, val_cfg):

	Dataset_train = get_dataset(training_cfg['dataset']['name'])
	Dataset_val = get_dataset(val_cfg['dataset']['name'])
	datasets = dict(train=Dataset_train, val=Dataset_val)

	batch_size = dict(train=training_cfg['batch_size'], val=val_cfg['batch_size'])

	_id_list_path = os.path.join(partition, '{}.txt')
	dataset_params = dict(train=dict(training_cfg['dataset']['params']), 
						  val=dict(val_cfg['dataset']['params']),
						  )
	dataset_params['train'].update(id_list_path=_id_list_path.format('train'))
	dataset_params['val'].update(id_list_path=_id_list_path.format('val'))

	datasets_dict = {x: datasets[x](**dataset_params[x]) for x in ['train', 'val']}
	dataloaders_dict = {x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size[x],  shuffle=(x == "train"), num_workers=0) for x in ['train', 'val']}

	return dataloaders_dict

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

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--dataset', type=str, default='APR_TAX_RWY')
	parser.add_argument('--partitions', nargs='+', type=int)
	parser.add_argument('--num_epochs', type=int, default=1)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


if __name__ == "__main__":

	args = parse_args()
	exper_name = os.path.basename(args.config).split(".")[0]
	dataset_dir = os.path.join('list', args.dataset)

	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	for glob in Path(os.path.join(dataset_dir)).glob("*"):

		partition_dir = str(glob)
		partition_number = int(glob.parts[-1].split("_")[-1])

		if args.partitions is not None:
			if partition_number not in args.partitions:
				continue

		partition = os.path.join(dataset_dir, 'partition_{}').format(partition_number)

		model_train = get_model(num_classes, training_cfg["model"]).to(device)
		model_val = get_model(num_classes, val_cfg["model"]).to(device)
		models = dict(train=model_train, val=model_val)
		
		criterion = custom_loss

		params_to_update = model_train.parameters()
		optimizer = optim.SGD(params_to_update, lr=0.0005, momentum=0.9, weight_decay=1e-5)
		scheduler = get_scheduler(training_cfg['scheduler']['name'])(optimizer, **training_cfg['scheduler']["params"])

		dataloaders = get_dataloaders(partition, training_cfg, val_cfg)

		checkpoint_dir = os.path.join('checkpoint', args.dataset, 'partition_{}', exper_name).format(partition_number)
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if last_checkpoint_path is not None:
			last_checkpoint = torch.load(last_checkpoint_path)
			model_train.load_state_dict(last_checkpoint["model_state_dict"])
			epoch = last_checkpoint["epoch"]
			optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
		else:
			epoch = 0

		results_dir = os.path.join('results', args.dataset, 'partition_{}', exper_name).format(partition_number)
		saver = Saver(num_classes, val_cfg['dataset']['params']['root'], results_dir, checkpoint_dir, epoch)

		train_model(models, dataloaders, criterion, optimizer, scheduler, device, training_cfg, val_cfg, num_classes, num_epochs=args.num_epochs, saver=saver)