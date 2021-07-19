import os
import torch

from core.metric.accuracy import AccuracyMetric

from core.config import args
from core.utils import logger, set_reproducible
from core.utils import device

from core.ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3


import torchvision.transforms as transforms

from core.dataset.imagenet.imagenet_folder import ImageFolder

if __name__ == "__main__":
    logger.info(args)
    set_reproducible(args.seed)

    # prepare data
    traindir = os.path.join(args.path, 'train')
    valdir = os.path.join(args.path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]), 
        class_num=1000, ratio=1.0
        )

    logger.info(f"val set length is {len(val_dataset)}")
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    ofa_supernet = OFAMobileNetV3(
        dropout_rate=0.1,
        width_mult=1.0,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4]
    )
    if args.eval_model == "adaxpert-100":
        arch = {'ks': [3,5,5,5,5,5,5,5,5,5,7,5,3,5,5,5,5,7,5,5], 'ratios': [6,6,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3], 'depths': [4,3,3,3,3]} 
    elif args.eval_model == "adaxpert-200":
        arch = {'ks': [5,7,5,7,7,7,7,7,5,5,7,7,7,7,7,7,7,7,7,7], 'ratios': [6,4,4,4,4,4,4,3,4,4,4,4,6,4,4,4,4,4,4,4], 'depths': [3,3,3,3,3]} 
    elif args.eval_model == "adaxpert-1000":
        arch = {'ks': [7,3,7,3,7,7,5,7,7,5,5,7,5,7,7,7,7,7,7,5], 'ratios': [6,6,6,3,4,6,6,6,4,4,6,4,4,4,6,4,6,6,6,4], 'depths': [4,4,3,3,3]} 
    else:
        assert False

    ofa_supernet.set_active_subnet(ks=arch['ks'], e=arch['ratios'], d=arch['depths'])
    model = ofa_supernet.get_active_subnet(preserve_weight=False)
    # TODO
    if args.pretrained_submodel_path:
        checkpoint = torch.load(args.pretrained_submodel_path)
        model.load_state_dict(checkpoint, map_location="cpu")
        # model.load_state_dict(checkpoint['state_dict'], map_location="cpu")

    model.cuda()
    model.eval()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    for i, (images, targets) in enumerate(val_loader):
        images, targets = images.cuda(), targets.cuda()
        
        with torch.no_grad():
            outputs = model(images)
        accuracy_metric.update(targets, outputs)
        acc1 = accuracy_metric.accuracy(1).rate
        acc5 = accuracy_metric.accuracy(5).rate
        logger.info(f"current top1 acc is {acc1:.4f} and current top5 acc is {acc5:.4f}")
