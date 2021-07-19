# Based on Once-for-All: eval_ofa_net.py
# Author Huberyniu, 2020.12.30

import os
import torch
import argparse
import torch.nn as nn

from core.dataset.seq2arch.mobilespace import str2arch

from core.spos.spos_mobilenet import SPOSMobileNet
from core.metric.accuracy import AccuracyMetric
from core.utils import logger

from core.utils import compute_flops, compute_nparam
from core.utils import device


class RewardFnSPOS(nn.Module):
    def __init__(self, args=None, val_loader=None, supernet_path=""):
        super(RewardFnSPOS, self).__init__()
        if args.gpu == 'all':
            device_list = range(torch.cuda.device_count())
            args.gpu = ','.join(str(_) for _ in device_list)
        else:
            device_list = [int(_) for _ in args.gpu.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        self.spos_network = SPOSMobileNet(n_classes=1000, depth_list=[2,3,4]).cuda()
        if len(supernet_path) > 0 :
            checkpoint_path = supernet_path
            logger.info(f"checkpoint path is {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.spos_network.load_state_dict(checkpoint['model'])
        self.val_loader = val_loader

    def forward(self, sample_arch, pre_arch_acc=0, pre_arch_flops=0):
        top1 = self.get_arch_acc(sample_arch)
        return top1 - pre_arch_acc

    def get_arch_acc(self, arch, batch_num=1):
        arch = str2arch(arch[0])
        self.spos_network.eval()
        accuracy_metric = AccuracyMetric(topk=(1, 5))
        for i, (images, targets) in enumerate(self.val_loader):
            images, targets = images.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = self.spos_network(images, arch)
            accuracy_metric.update(targets, outputs)
            if i == int(batch_num-1):
                break
        top1 = accuracy_metric.accuracy(1).rate
        top5 = accuracy_metric.accuracy(5).rate
        logger.info(f'Results are: top1=%.4f,\t top5=%.4f' % (top1, top5))
        return top1, top5

    def get_arch_params_madds(self, arch):
        arch = str2arch(arch[0])
        sub_model = self.spos_network.get_subnet(arch)
        
        # flops & params & structure
        n_params = compute_nparam(sub_model, skip_pattern="auxiliary")
        flops = compute_flops(sub_model, (1, 3, 224, 224), skip_pattern="auxiliary", device=device)
        logger.info(f"n_params={n_params/1000**2:.2f}M, "
                f"Madds={flops/1000**2:.2f}M.")
        
        return n_params/1000**2, flops/1000**2

    def get_arch_reward_details(self, arch, batch_num=1):
        # logger.info(f"to query the arch {arch[0]}")
        top1, top5 = self.get_arch_acc(arch, batch_num=batch_num)
        n_params, flops = self.get_arch_params_madds(arch)

        return top1, top5, n_params, flops

    def arch_distance(self, arch1, arch2):
        arch1, arch2 = str2arch(arch1[0]), str2arch(arch2[0])

        arch1_bool = []
        index = [0, 0, 0, 0]
        for de in arch1.depths:
            for i in range(de):
                index[i] = 1
            arch1_bool = arch1_bool + index
            index = [0, 0, 0, 0]
        
        arch2_bool = []
        index = [0, 0, 0, 0]
        for de in arch2.depths:
            for i in range(de):
                index[i] = 1
            arch2_bool = arch2_bool + index
            index = [0, 0, 0, 0]

        for i in range(len(arch1_bool)):
            if arch1_bool[i] == 0:
                arch1.ks[i] = 0
                arch1.ratios[i] = 0
            if arch2_bool[i] == 0:
                arch2.ks[i] = 0
                arch2.ratios[i] = 0
        
        ks_distance = list(map(lambda x: x[0]-x[1], zip(arch2.ks, arch1.ks)))
        ratios_distance = list(map(lambda x: x[0]-x[1], zip(arch2.ratios, arch1.ratios)))

        ksd_new = [i for i in ks_distance if i != 0]
        ratios_new = [i for i in ratios_distance if i != 0]

        arch_distance = len(ksd_new)+len(ratios_new)
        logger.info(f"distance between archs={arch_distance:.1f}")

        return arch_distance

