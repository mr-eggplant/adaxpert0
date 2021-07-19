import os
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import argparse

from core.controller import MBSpaceController

from core.engine.policy_learning import train_controller
from core.config import args
from core.utils import logger, set_reproducible
from core.utils import device

from core.model.rewardfnspos import RewardFnSPOS

import torchvision.transforms as transforms

from core.dataset.imagenet.imagenet_folder import ImageFolder
from core.metric.distribution_distance import calculate_distribution_distance

if __name__ == "__main__":
    logger.info(args)
    set_reproducible(args.seed)
    writer = tensorboard.SummaryWriter(args.output)

    # prepare data: spliting ImageNet
    traindir = os.path.join(args.path, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]), 
        class_num=args.dataset_class_num, 
        ratio=args.dataset_ratio, flag='val', split=args.dataset_split)
    logger.info(f"val set length is {len(val_dataset)}")
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    """
    compute the distribution distance between new data and current data.
    mu_old/mu_new is the mean vector of data_old/data_new; sigma_old/sigma_new is the covariance matrix of data_old/data_new.
    the users need to first compute {mu_old, mu_new, sigma_old, sigma_new} based on their own model and data.
    default value of dd is 0.
    """

    mu_old, sigma_old, mu_new, sigma_new = None, None, None, None
    dd = calculate_distribution_distance(mu_old, sigma_old, mu_new, sigma_new) 

    # dd = 1.0

    # train controller
    pre_arch_seq = [args.previous_arch] 
    logger.info(f"pre_arch_seq is {pre_arch_seq}")
    rewardfn = RewardFnSPOS(args=args, val_loader=val_loader, supernet_path=args.supernet_path)
    controller = MBSpaceController(n_conditions=100, device=device).to(device=device)
    if args.previous_controller:
        logger.info("loding previous controller ing...")
        controller_base = torch.load(args.previous_controller)
        controller.load_state_dict(controller_base)

    controller_optimizer = optim.Adam(controller.parameters(), args.controller_lr,
                                      betas=(0.5, 0.999), weight_decay=5e-4)
    train_controller(max_iter=args.pl_iters, entropy_coeff=2e-4, grad_clip=args.controller_grad_clip,
                     controller=controller, rewardfn=rewardfn, optimizer=controller_optimizer, writer=writer, 
                     search_space="mobileblock", previous_arch_seq=pre_arch_seq, data_distance=dd, r_lambda=args.r_lambda)

