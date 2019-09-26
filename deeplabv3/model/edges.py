from .deeplab import DeepLabDecoder1, _Deeplabv3Plus
from torch.nn import functional as F

class EdgesModel(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3Plus1, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.edges_classifier = nn.Conv2d(256, 1, 1, 1, 0, 1, bias=False)
		init_conv(self.edges_classifier)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x_edges = self.edges_classifier(x)
		x = self.classifier(x)
		
		if self.training:
			result = OrderedDict()
			result["out"] = x
			result["out_edges"] = x_edges
			if self.aux_clf is not None:
				x = features["aux"]
				x = self.aux_clf(x)
				x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
				result["aux"] = x
			return result
		else:
			return self.predict(x)

