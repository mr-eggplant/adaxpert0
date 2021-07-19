# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch.nn as nn

from core.ofa.utils import MyNetwork, MyGlobalAvgPool2d

__all__ = ['MobileNet']


class MobileNet(MyNetwork):

	def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):
		super(MobileNet, self).__init__()

		self.first_conv = first_conv
		self.blocks = nn.ModuleList(blocks)
		self.final_expand_layer = final_expand_layer
		self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
		self.feature_mix_layer = feature_mix_layer
		self.classifier = classifier

	def forward(self, x):
		x = self.first_conv(x)
		for block in self.blocks:
			x = block(x)
		x = self.final_expand_layer(x)
		x = self.global_avg_pool(x)  # global average pooling
		x = self.feature_mix_layer(x)
		feature = x.view(x.size(0), -1)
		x = self.classifier(feature)
		return feature, x

	def set_bn_param(self, bn_momentum=0.1, bn_eps=0.00001):
		for m in self.modules():
			if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
				m.momentum = bn_momentum
				m.eps = bn_eps
		return