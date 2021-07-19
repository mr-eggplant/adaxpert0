# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
import torch.nn as nn

from core.ofa.utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock
from core.ofa.utils import make_divisible, val2list, MyNetwork
from core.spos.mobilenet import MobileNet

__all__ = ['SPOSMobileNet']


class SPOSMobileNet(nn.Module):

	def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None, width_mult=1.0,
				 ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4]):
		super(SPOSMobileNet, self).__init__()
		self.width_mult = width_mult
		self.ks_list = val2list(ks_list, 1)
		self.expand_ratio_list = val2list(expand_ratio_list, 1)
		self.depth_list = val2list(depth_list, 1)

		self.ks_list.sort()
		self.expand_ratio_list.sort()
		self.depth_list.sort()     

		self.bn_param = bn_param                          

		base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

		final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		stride_stages = [1, 2, 2, 2, 1, 2]
		act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, False, True, False, True, True]
		n_block_list = [1] + [max(self.depth_list)] * 5
		width_list = []
		for base_width in base_stage_width[:-2]:
			width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
			width_list.append(width)

		input_channel, first_block_dim = width_list[0], width_list[1]
		# first conv layer
		self.first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, act_func='h_swish')
		first_block_conv = MBConvLayer(
			in_channels=input_channel, out_channels=first_block_dim, kernel_size=3, stride=stride_stages[0],
			expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
		)
		self.first_block = ResidualBlock(
			first_block_conv,
			IdentityLayer(first_block_dim, first_block_dim) if input_channel == first_block_dim else None,
		)

		# inverted residual blocks
		self.block_group_info = []
		self.blocks = nn.ModuleList([])
		_block_index = 0
		feature_dim = first_block_dim

		for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
													   stride_stages[1:], act_stages[1:], se_stages[1:]):
			self.block_group_info.append([_block_index + i for i in range(n_block)])
			_block_index += n_block

			output_channel = width
			for i in range(n_block):
				if i == 0:
					stride = s
				else:
					stride = 1
				this_layer = nn.ModuleList([])
				for ks in self.ks_list:
					for r in self.expand_ratio_list:
						mobile_inverted_conv = MBConvLayer(
							in_channels=feature_dim, out_channels=output_channel,
							kernel_size=ks, stride=stride, expand_ratio=r,
							act_func=act_func, use_se=use_se
							)
						if stride == 1 and feature_dim == output_channel:
							shortcut = IdentityLayer(feature_dim, feature_dim)
						else:
							shortcut = None
						this_layer.append(ResidualBlock(mobile_inverted_conv, shortcut))
				self.blocks.append(this_layer)
				feature_dim = output_channel
		
		# final expand layer, feature mix layer & classifier
		self.final_expand_layer = ConvLayer(feature_dim, final_expand_width, kernel_size=1, act_func='h_swish')
		self.feature_mix_layer = ConvLayer(
			final_expand_width, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
		)

		self.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

		# set bn param
		self.set_bn_param(bn_momentum=bn_param[0], bn_eps=bn_param[1])

		# runtime_depth
		self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]


	def forward(self, x, arch):
		# first conv
		self.set_runtime_depth(arch.depths)
		x = self.first_conv(x)
		# first block
		# x = self.blocks[0](x)
		x = self.first_block(x)
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				# x = self.blocks[idx](x)
				c_ks, c_r = arch.ks[idx], arch.ratios[idx]
				idx_layer = self.ks_list.index(c_ks) * len(self.ks_list) + self.expand_ratio_list.index(c_r)
				x = self.blocks[idx][idx_layer](x)
		x = self.final_expand_layer(x)
		x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
		x = self.feature_mix_layer(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def set_runtime_depth(self, d=None, **kwargs):
		depth = val2list(d, len(self.block_group_info))

		for i, d in enumerate(depth):
			if d is not None:
				self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

	def set_bn_param(self, bn_momentum=0.1, bn_eps=0.00001):
		for m in self.modules():
			if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
				m.momentum = bn_momentum
				m.eps = bn_eps
		return

	def get_subnet(self, arch):
		self.set_runtime_depth(arch.depths)

		first_conv = copy.deepcopy(self.first_conv)
		blocks = [copy.deepcopy(self.first_block)]

		final_expand_layer = copy.deepcopy(self.final_expand_layer)
		feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
		classifier = copy.deepcopy(self.classifier)
		# classifier = LinearLayer(1280, 1000)

		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				c_ks, c_r = arch.ks[idx], arch.ratios[idx]
				idx_layer = self.ks_list.index(c_ks) * len(self.ks_list) + self.expand_ratio_list.index(c_r)
				blocks.append(copy.deepcopy(self.blocks[idx][idx_layer]))

		_subnet = MobileNet(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
		_subnet.set_bn_param(bn_momentum=self.bn_param[0], bn_eps=self.bn_param[1])

		return _subnet
