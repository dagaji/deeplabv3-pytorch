import yaml
import pdb
from collections import OrderedDict
import os

# def list_to_od(_list):
# 	od = OrderedDict()
# 	for el in _list:
# 		key = str(*el.keys())
# 		value = el[key]
# 		if isinstance(value, list):
# 			od[key] = list_to_od(value)
# 		else:
# 			od[key] = value
# 	return od

def get_cfgs(config_path):

	exper_name = os.path.basename(config_path).split('.')[0]

	with open(config_path, 'r') as stream:
	    try:
	        config = yaml.safe_load(stream)
	        num_classes = config["num_classes"]
	        training_cfg = config["training"]
	        val_cfg = config["validation"]
	        return num_classes, training_cfg, val_cfg
	    except yaml.YAMLError as exc:
	        print(exc)
	return None

def get_cfgs_video(config_path):

	exper_name = os.path.basename(config_path).split('.')[0]

	with open(config_path, 'r') as stream:
	    try:
	        config = yaml.safe_load(stream)
	        num_classes = config["num_classes"]
	        video_cfg = config["video"]
	        return num_classes, video_cfg
	    except yaml.YAMLError as exc:
	        print(exc)
	return None



# if __name__ == "__main__":
# 	with open("config/resnet101.yml", 'r') as stream:
# 	    try:
# 	        config = yaml.safe_load(stream)
# 	        if 'augmentations' in config['dataset']['train']:
# 	        	config['dataset']['train']['augmentations'] = list_to_od(config['dataset']['train']['augmentations'])
# 	        pdb.set_trace()
# 	    except yaml.YAMLError as exc:
# 	        print(exc)