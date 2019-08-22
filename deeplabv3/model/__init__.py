from .deeplab import *
from .predict import *
from .instance import *
from .ori import *
from .ori_v2 import *
import torch.nn as nn

models_map = {'deeplabv3' : Deeplabv3,
			  'deeplabv3+1' : Deeplabv3Plus1,
			  'deeplabv3+2' : Deeplabv3Plus2,
			  'deeplabv3+instance1': Deeplabv3PlusInstance1,
			  'deeplabv3+instance2': Deeplabv3PlusInstance2,
			  'deeplabv3+instance3': Deeplabv3PlusInstance3,
			  'deeplabv3+ori': Deeplabv3PlusOri,
			  'deeplabv3+ori2': Deeplabv3PlusOri2,
			  'deeplabv3+ori3': Deeplabv3PlusOri3,
			  # 'deeplabv3+ori_v2': Deeplabv3PlusOri_v2,
			  # 'hist_v2': Deeplabv3PlusHist_v2,
			  'hist': Deeplabv3PlusHist,
			  'angle_clf': Deeplabv3PlusAngleClf,
			  }

predict_map = {'multilabel': multilabel_predict,
			   'erode': erode_predict,
			   'default': argmax_predict,
			   'argmax_cc' : argmax_cc_predict}

output_stride_params = { 16: dict(replace_stride_with_dilation=[False, False, True], rates=[6, 12, 18]),
						 8:  dict(replace_stride_with_dilation=[False, True, True],  rates=[12, 24, 36]),
						 4 : dict(replace_stride_with_dilation=[True, True, True],   rates=[24, 48, 72]),
						}

def get_model(n_classes, cfg, aux=False):

	return_layers = dict(layer4='out', layer1='skip1', layer3='aux')

	kw_backbone_args = dict(output_stride_params[cfg['stride']])
	kw_backbone_args.update(return_layers=return_layers)
	pretrained_model = load_pretrained_model(kw_backbone_args)
	
	predict_key = cfg.get('predict', 'default')
	out_planes_skip = cfg.get('out_planes_skip', 48)
	model = models_map[cfg['name']](n_classes, 
									pretrained_model, 
									predict_map[predict_key], 
									aux=aux, 
									out_planes_skip=out_planes_skip)
	if 'dilation' in cfg:
		_dilate_conv = lambda m : dilate_conv(m, cfg['dilation'])
		model.apply(_dilate_conv)
	return model


def dilate_conv(m, dilation_factor):
	if isinstance(m, nn.Conv2d):
		if m.kernel_size[0] > 1:
			dilation = np.array(m.dilation) * dilation_factor
			m.dilation = tuple(dilation.tolist())
			pading = dilation if m.kernel_size[0] == 3 else 3 * dilation
			m.padding = tuple(pading.tolist())
	elif isinstance(m, nn.MaxPool2d):
		m.kernel_size = m.kernel_size * dilation_factor + 1
		m.padding = int(m.kernel_size / 2)
