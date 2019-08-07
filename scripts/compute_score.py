import os.path
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import pdb
import json

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--exper_name', type=str, required=True)
	parser.add_argument('--stride', type=int, default=4)
	parser.add_argument('--results_dir', type=str, default='../results/APR_TAX_RWY/')
	parser.add_argument('-cc', dest='do_cc', action='store_true')
	parser.set_defaults(do_cc=False)
	return parser.parse_args()

args = parse_args()

scores_list = []
for partition_glob in Path(args.results_dir).glob("*"):
	for exper_glob in partition_glob.glob("*"):
		exper_name = os.path.basename(str(exper_glob))
		if exper_name == args.exper_name:
			for val_exper_glob in exper_glob.glob("*"):
				val_exper_name = os.path.basename(str(val_exper_glob))
				if val_exper_name == 'res1000_s{}'.format(args.stride) + ('_cc' if args.do_cc else ''):
					with open(os.path.join(str(val_exper_glob), 'score.json')) as json_file:
						score_dict = json.load(json_file)["scores"][-1]
						per_class_iu = np.array(score_dict[score_dict.keys()[0]])
						per_class_iu[np.isnan(per_class_iu)] = 0.0
						scores_list.append(per_class_iu)
						
if len(scores_list) > 0:
	scores_array = np.array(scores_list)
	pre_class_iu_total = scores_array.sum(0) / (scores_array > 0).sum(0)
	print(scores_array)
	print("pre_class_iu_total={}".format(pre_class_iu_total))
	print("miou_total={}".format(pre_class_iu_total.mean()))