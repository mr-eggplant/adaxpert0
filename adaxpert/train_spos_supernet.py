import os
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import argparse

from core.controller import MBSpaceController

from core.config import args
from core.utils import logger, set_reproducible
from core.utils import device

from core.dataset.seq2arch.mobilespace import str2arch

from core.dataset.imagenet.imagenet_folder import ImageFolder
from core.spos.spos_mobilenet import SPOSMobileNet
from core.metric.accuracy import AccuracyMetric

import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.datasets as dset

import horovod.torch as hvd


if __name__ == "__main__":
    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())
    
    num_gpus = hvd.size()

    if hvd.rank() == 0:
        logger.info(f"num_gpus is {num_gpus}")

    logger.info(args)
    set_reproducible(args.seed)
    writer = tensorboard.SummaryWriter(args.output)
    
    # prepare data: splitng ImageNet
    traindir = os.path.join(args.path, 'train')
    valdir = os.path.join(args.path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]), class_num=args.dataset_class_num, 
        ratio=args.dataset_ratio, flag='train', split=args.dataset_split)

    val_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), class_num=args.dataset_class_num, 
        ratio=args.dataset_ratio, flag='val', split=args.dataset_split)

    test_dataset = ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]), 
        class_num=args.dataset_class_num, ratio=1.0
        )

    if hvd.rank() == 0:
        logger.info(f"train set length is {len(train_dataset)}")
        logger.info(f"val set length is {len(val_dataset)}")
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_gpus, rank=hvd.rank())
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=num_gpus, rank=hvd.rank())
        val_sampler = None
        test_sampler = None
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)
    
    print(device)
    model = SPOSMobileNet(n_classes=1000, depth_list=[2,3,4])
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9, 0.0005)

    # resume training
    if args.supernet_resume_train:
        if hvd.rank() == 0:
            logger.info("loading checkpoint...")
        checkpoint_path = args.supernet_resume_path
        logger.info(f"loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    controller = MBSpaceController(device=device).cuda()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_parameters(controller.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if hvd.rank() == 0:
        logger.info(model)

    with torch.no_grad():
        *arch_seq, _, _ = controller(force_uniform=True)
        arch = str2arch(arch_seq[0])


    for epoch in range(args.supernet_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        if hvd.rank() == 0:
                logger.info(f"----------------------current epoch is {epoch + 1}---------------------")

        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.cuda(), targets.cuda()
            if i % args.num_batches_each_arch == 0:
                with torch.no_grad():
                    *arch_seq, _, _ = controller(force_uniform=True)
                    arch = str2arch(arch_seq[0])
            outputs = model(images, arch)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()

        
        model.eval()
        for i, (images, targets) in enumerate(val_loader):
            images, targets = images.cuda(), targets.cuda()
            accuracy_metric = AccuracyMetric(topk=(1, 5))
            with torch.no_grad():
                *arch_seq, _, _ = controller(force_uniform=True)
                arch = str2arch(arch_seq[0])
                outputs = model(images, arch)
            accuracy_metric.update(targets, outputs)
            acc = accuracy_metric.accuracy(1).rate
            if hvd.rank() == 0:
                logger.info(f"sampled arch is {arch}, and its test acc is {acc:.4f}")
        

        if (epoch+1) % args.save_freq_of_supernet == 0:
            if hvd.rank() == 0:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                # torch.save(model.state_dict(), os.path.join(args.output, str(epoch+1)+'net_params.pt'))
                torch.save(state, os.path.join(args.output, str(epoch+1)+'checkpoint.pt'))
