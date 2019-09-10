from .deeplab import *
from .predict import *
from .hist import *
import torch.nn as nn

models_map = {'deeplabv3' : Deeplabv3,
			  'deeplabv3+1' : Deeplabv3Plus1,
			  'deeplabv3+2' : Deeplabv3Plus2,
			  'hist': Deeplabv3PlusHist,
			  }

predict_map = {'argmax': argmax_predict,
			   'line_dect': line_detection}

output_stride_params = { 16: dict(replace_stride_with_dilation=[False, False, True], rates=[6, 12, 18]),
						 8:  dict(replace_stride_with_dilation=[False, True, True],  rates=[12, 24, 36]),
						 4 : dict(replace_stride_with_dilation=[True, True, True],   rates=[24, 48, 72]),
						}

def get_model(n_classes, cfg, aux=False):

	return_layers = dict(layer4='out', layer1='skip1', layer3='aux')

	kw_backbone_args = dict(output_stride_params[cfg['stride']])
	kw_backbone_args.update(return_layers=return_layers)
	pretrained_model = load_pretrained_model(kw_backbone_args)
	
	predict_key = cfg.get('predict', 'argmax')
	out_planes_skip = cfg.get('out_planes_skip', 48)
	model = models_map[cfg['name']](n_classes, 
									pretrained_model, 
									predict_map[predict_key], 
									aux=aux, 
									out_planes_skip=out_planes_skip)
	return model
