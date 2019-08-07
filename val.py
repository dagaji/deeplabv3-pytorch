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
from argparse import ArgumentParser
from torchsummary import summary
from deeplabv3.save import ResultsSaverFactory, CheckpointSaver
from pathlib import Path
import time

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('diff_time = {}'.format(te-ts))
        return result

    return timed

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

def get_dataloader(id_list_path, dataset_cfg, batch_size, shuffle=True):

	dataset_params = dict(dataset_cfg['params'])
	dataset_params.update(id_list_path=id_list_path)
	dataset = get_dataset(dataset_cfg['name'])(**dataset_params)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--dataset', type=str, default='APR_TAX_RWY')
	parser.add_argument('--partitions', nargs='+', type=int)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


@timeit
def validate(val_model, val_loader, num_classes, device, saver=None):

	val_model.eval()   # Set model to evaluate mode

	running_score = RunningScore(num_classes)
	running_score.reset()

	with torch.set_grad_enabled(False):

		# Iterate over data.
		for _iter, data in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):

			image_id = data['image_id']
			inputs = data['image'].to(device)
			labels = data['label'].to(device)

			preds = val_model(inputs)

			running_score.update(labels, preds)
			if saver is not None:
				saver.save_vis(image_id, preds.cpu().numpy())

	per_class_iu = running_score.get_pre_class_iu()
	print('Per_class IoU: {}'.format(per_class_iu))
	if saver is not None:
		saver.save_score(per_class_iu)

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
		print('>> Partition {}'.format(partition_number))
		print()

		partition = os.path.join(dataset_dir, 'partition_{}').format(partition_number)
		_id_list_path = os.path.join(partition, '{}.txt')

		model_train = get_model(num_classes, training_cfg["model"], training_cfg['aux_loss']).to(device)

		val_expers = {}
		for _val_exper in val_cfg['val_expers']:
			model_val = get_model(num_classes, _val_exper["model"]).to(device)
			val_dataloader = get_dataloader(_id_list_path.format('val'), _val_exper['dataset'], val_cfg['batch_size'], shuffle=False)
			val_expers[_val_exper['name']] = dict(model_val=model_val, val_dataloader=val_dataloader)

		checkpoint_dir = os.path.join('checkpoint', args.dataset, 'partition_{}', exper_name).format(partition_number)
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if last_checkpoint_path is not None:
			last_checkpoint = torch.load(last_checkpoint_path)
			model_train.load_state_dict(last_checkpoint["model_state_dict"])
			current_epoch = last_checkpoint["epoch"]
		else:
			current_epoch = 0

		results_dir = os.path.join('results', args.dataset, 'partition_{}', exper_name).format(partition_number)
		saver_factory = ResultsSaverFactory(num_classes, results_dir, current_epoch)

		for val_exper_name, val_exper in val_expers.items():
			print('>> {}'.format(val_exper_name))
			val_model, val_dataloader = val_exper['model_val'], val_exper['val_dataloader']
			root_folder = val_dataloader.dataset.root
			results_saver = saver_factory.get_saver(val_exper_name, 
																root_folder, 
																0)
			val_model.load_state_dict(model_train.state_dict(), strict=False)
			validate(val_model, 
				val_dataloader, 
				num_classes, 
				device, saver=results_saver)
			print()

