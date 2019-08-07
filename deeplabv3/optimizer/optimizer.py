from torch.optim import SGD
import pdb
import torch.nn as nn

optimizers = {"sgd": SGD}

def get_params(modules_list):
	for m in modules_list:
		if isinstance(m[1], nn.Conv2d):
			for p in m[1].parameters():
				yield p

def filter_modules(all_modules, modules_name):
	extracted_modules = []
	for idx, m in enumerate(all_modules):
		if m[0] in modules_name:
			extracted_modules.append(all_modules.pop(idx))
	return extracted_modules


def get_optimizer(model, cfg):

	if cfg['name'] not in optimizers:
		raise NotImplementedError("Optimizer {} not implemented".format(cfg['name']))

	all_modules = list(model.named_modules())
	params_groups = []
	for group_name in cfg['groups']:
		group_info = cfg['groups'][group_name]
		group_modules = filter_modules(all_modules, group_info.pop('modules'))
		group_dict = dict(params=get_params(group_modules), **group_info)
		params_groups.append(group_dict)
	params_groups.append(dict(params=get_params(all_modules)))
	cfg.pop('groups')

	return optimizers.get(cfg.pop('name'))(params=params_groups, **cfg)
