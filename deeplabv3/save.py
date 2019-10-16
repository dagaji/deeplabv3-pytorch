from .vis import make_palette, vis_seg
import os.path
import os
import cv2
import numpy as np
import pdb
import json
import torch
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def makedir(_dir):
	if not os.path.exists(_dir):
		os.makedirs(_dir)

class ResultsSaverFactory:
	def __init__(self, n_classes, save_dir, current_epoch):
		self.n_classes = n_classes
		self.save_dir = os.path.join(save_dir, '{}')
		self.current_epoch = current_epoch + 1

	def get_saver(self, val_exper_name, root_dir, epoch):
		return ResultsSaver(self.n_classes,
			                root_dir,
			                self.save_dir.format(val_exper_name), self.current_epoch + epoch
			                )

class ResultsSaver:
	def __init__(self, n_classes, root_dir, save_dir, epoch):
		self.palette = make_palette(n_classes)
		self.imgs_dir = os.path.join(root_dir, 'images')
		self.save_dir_vis = os.path.join(save_dir, 'epoch_{}').format(epoch)
		makedir(self.save_dir_vis)
		self.score_path = os.path.join(save_dir, 'score.json')
		self.init_score_file()
		self.epoch = epoch

	def init_score_file(self):
		if not os.path.exists(self.score_path):
			os.mknod(self.score_path)
			scores = dict(scores=[])
			with open(self.score_path, 'w') as f:
				json.dump(scores, f)


	def save_vis(self, img_id, pred):

		img_id = img_id[0]
		img_path = os.path.join(self.imgs_dir, img_id)
		save_path = os.path.join(self.save_dir_vis, img_id)
		img = cv2.imread(img_path)
		# Descomentar cuando panos
		h, w = img.shape[:2]
		new_size = (int(w * 0.65), int(h * 0.65))
		img = cv2.resize(img, new_size)
		vis_img = vis_seg(img, np.squeeze(pred), self.palette)
		# plt.figure()
		# plt.imshow(vis_img[...,::-1])
		# plt.show()
		cv2.imwrite(save_path, vis_img)

	def save_score(self, per_class_iu):

		with open(self.score_path, 'r') as f:
			scores = json.load(f)

		scores_list = scores['scores']
		scores_list.append({str(self.epoch) : per_class_iu.tolist()})
		scores.update(scores=scores_list)

		with open(self.score_path, 'w') as f:
			json.dump(scores, f)

		
class CheckpointSaver:
	def __init__(self, checkpoint_dir, start_epoch):
		self.checkpoint_dir = checkpoint_dir
		makedir(self.checkpoint_dir)
		self.start_epoch = start_epoch + 1


	def save_checkpoint(self, epoch, model, optimizer):

		epoch += self.start_epoch
		checkpoint_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch))
		torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

class VideoSaver:

	def __init__(self, n_classes, save_path, fps=10.0):
		self.palette = make_palette(n_classes)
		self.save_path = save_path
		self.fps = int(fps)
		self.out = None
		
	def save_frame(self, img, pred):
		img = np.squeeze(img)
		pred = np.squeeze(pred)
		if self.out is None:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			self.out = cv2.VideoWriter(self.save_path, fourcc, self.fps, img.shape[:2][::-1])
		# plt.figure()
		# plt.imshow(vis_seg(img, pred, self.palette))
		# plt.show()
		self.out.write(vis_seg(img, pred, self.palette))

	def save_video(self):
		if self.out is not None:
			self.out.release()
			self.out = None




