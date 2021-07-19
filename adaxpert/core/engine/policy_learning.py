
import os
import copy
import random
import functools

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard


from core.metric import AverageMetric, MovingAverageMetric
from core.controller import MBSpaceController
from core.config import args
from core.utils import *

from core.model.rewardfnspos import RewardFnSPOS

from core.dataset.seq2arch.mobilespace import str2arch
from core.dataset.tensorize.mobilespace import mb_arch2tensor


def single_batchify(*items):
    return [item.unsqueeze(0) for item in items]


best_iters = -1


def train_controller(max_iter: int, entropy_coeff: float, grad_clip: int,
                     controller: MBSpaceController, rewardfn: RewardFnSPOS,
                     optimizer: optim.Optimizer, writer: tensorboard.SummaryWriter,
                     log_frequence: int = 10, search_space=None, previous_arch_seq=None,
                     data_distance=0, r_lambda=1e-5):
    controller.train()
    optimizer.zero_grad()

    policy_loss_avg = MovingAverageMetric()
    entropy_mavg = MovingAverageMetric()
    logp_mavg = MovingAverageMetric()
    score_avg = MovingAverageMetric()
    top1_avg = MovingAverageMetric()
    flops_avg = MovingAverageMetric()
    arch_dis_avg = MovingAverageMetric()

    pre_top1, _, _, pre_flops = rewardfn.get_arch_reward_details(previous_arch_seq)

    pre_arch = str2arch(previous_arch_seq[0])
    pre_arch_tensor = mb_arch2tensor(pre_arch)
    dd = data_distance
    # embedding for dd:
    dds = [i*0.05 for i in range(100)]
    dd_diff = [abs(i-dd) for i in dds]
    index = dd_diff.index(min(dd_diff))
    condition_tensor = torch.tensor([index], dtype=torch.long)

    for iter_ in range(max_iter):

        logger.info("---------------------------sample one arch------------------------------------")

        # *arch_seq, logp, entropy = controller()
        *arch_seq, logp, entropy = controller(condition_tensor, pre_arch_tensor)

        logger.info(f"sampled arch is {arch_seq}")

        with torch.no_grad():
            # sample_arch = [tensorize_fn(seq2arch_fn(arch_seq), device=device)]
            top1, _, _, flops = rewardfn.get_arch_reward_details(arch_seq)
            distance = rewardfn.arch_distance(previous_arch_seq, arch_seq)
            if dd == 0:
                score = top1 - pre_top1
            else:
                score = (top1 - pre_top1) - (r_lambda/dd) * (max(0, flops - pre_flops))

        policy_loss = -logp * score - entropy_coeff * entropy

        optimizer.zero_grad()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
        policy_loss.backward()
        optimizer.step()

        policy_loss_avg.update(policy_loss)
        entropy_mavg.update(entropy)
        logp_mavg.update(logp)
        score_avg.update(score)
        top1_avg.update(top1)
        flops_avg.update(flops)
        arch_dis_avg.update(distance)

        if iter_ % log_frequence == 0:
            logger.info(
                ", ".join([
                    "Policy Learning",
                    f"iter={iter_:03d}",
                    f"policy loss={policy_loss_avg.compute():.4f}",
                    f"entropy={entropy_mavg.compute():.4f}",
                    f"logp={logp_mavg.compute():.4f}",
                ])
            )
            writer.add_scalar("policy_learning/loss", policy_loss_avg.compute(), iter_)
            writer.add_scalar("policy_learning/entropy", entropy_mavg.compute(), iter_)
            writer.add_scalar("policy_learning/logp", logp_mavg.compute(), iter_)
            writer.add_scalar("policy_learning/reward", score_avg.compute(), iter_)
            writer.add_scalar("policy_learning/arch_top1", top1_avg.compute(), iter_)
            writer.add_scalar("policy_learning/arch_flops", flops_avg.compute(), iter_)
            writer.add_scalar("policy_learning/arch_distance", arch_dis_avg.compute(), iter_)

        if (iter_ + 1) % args.evaluate_controller_freq == 0:
            torch.save(controller.state_dict(), os.path.join(args.output,f"controller-{iter_}.path"))

            for i in range(10):
                logger.info(f"---------------------{i}-th derive arch-------------------------")
                with torch.no_grad():
                    *arch_seq, logp, entropy = controller(condition_tensor, pre_arch_tensor)
                    logger.info(f"derived arch is {arch_seq}")
                    top1, _, _, flops = rewardfn.get_arch_reward_details(arch_seq, batch_num=100000)
                    distance = rewardfn.arch_distance(previous_arch_seq, arch_seq)