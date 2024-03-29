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


# class ResultsLogger:
# 	def __init__(self, save_dir):

# 		self.histogram_dir = os.path.join(save_dir, 'hist')
# 		self.vis_pre_dir = os.path.join(save_dir, 'pre-NMS')
# 		self.vis_post_dir = os.path.join(save_dir, 'post-NMS')
# 		self.saved_hist = False
# 		self.img_idx = 0
# 		self.root = '/home/davidgj/projects/APR_TAX_RWY/images'
# 		self.palette = make_palette(4)

# 		makedir(self.histogram_dir)
# 		makedir(self.vis_pre_dir)
# 		makedir(self.vis_post_dir)

# 	def save_hist(self, res, angles):

# 		if not self.saved_hist:
# 			for idx, _res in enumerate(res[0]):
# 				fig = plt.figure()
# 				save_path = os.path.join(self.histogram_dir, "angle_{}.png".format(angles[idx]))
# 				plt.imshow(_res.cpu().detach().numpy().squeeze(), vmax=1.0)
# 				fig.savefig(save_path)
# 			self.saved_hist = True

# 	def save_vis(self, img_id, mask, save_dir):
# 		img_id = img_id[0]
# 		img_path = os.path.join(self.root, img_id)
# 		img = cv2.imread(img_path)
# 		vis_img = vis_seg(img, np.squeeze(mask), self.palette)
# 		save_path = os.path.join(save_dir, "frame_{}.png".format(self.img_idx))
# 		cv2.imwrite(save_path, vis_img)

# 	def save_pre(self, img_id, mask):
# 		self.save_vis(img_id, mask, self.vis_pre_dir)

# 	def save_post(self, img_id, mask):
# 		self.save_vis(img_id, mask, self.vis_post_dir)
# 		self.img_idx += 1


# class ResultsSaverFactory:
# 	def __init__(self, n_classes, save_dir, current_epoch):
# 		self.n_classes = n_classes
# 		self.save_dir = os.path.join(save_dir, '{}')
# 		self.current_epoch = current_epoch + 1

# 	def get_saver(self, val_exper_name, root_dir, epoch):
# 		return ResultsSaver(self.n_classes,
# 			                root_dir,
# 			                self.save_dir.format(val_exper_name), self.current_epoch + epoch
# 			                )

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


	def __call__(self, pred, data):
		pred = pred.cpu().numpy().squeeze()
		img_id = data['image_id'][0]
		img_path = os.path.join(self.imgs_dir, img_id)
		save_path = os.path.join(self.save_dir_vis, img_id)
		img = cv2.imread(img_path)
		vis_img = vis_seg(img, pred, self.palette)
		cv2.imwrite(save_path, vis_img)

	# def save_score(self, per_class_iu):

	# 	with open(self.score_path, 'r') as f:
	# 		scores = json.load(f)

	# 	scores_list = scores['scores']
	# 	scores_list.append({str(self.epoch) : per_class_iu.tolist()})
	# 	scores.update(scores=scores_list)

	# 	with open(self.score_path, 'w') as f:
	# 		json.dump(scores, f)

		
class CheckpointSaver:
	def __init__(self, checkpoint_dir, start_epoch):
		self.checkpoint_dir = checkpoint_dir
		makedir(self.checkpoint_dir)
		self.start_epoch = start_epoch + 1


	def __call__(self, epoch, model, optimizer):

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




