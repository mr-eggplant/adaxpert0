import os
import argparse
import json

def get_config():
    parser = argparse.ArgumentParser("adaxpert")

    parser.add_argument('--output', type=str, default="", help='output directory')
    # delete seed for specnet
    parser.add_argument('--seed', type=int, default=2021, help='seed')

    parser.add_argument('--train_batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=100, help='validation batch size')

    parser.add_argument('--pl_iters', type=int, default=10000, help='num of training epochs for controller')

    parser.add_argument('--controller_grad_clip', type=float, default=None, help='')
    parser.add_argument('--entropy_coeff', type=float, default=5e-4, help='the coefficient of the controller')

    parser.add_argument('--n_sample_architectures', type=int, default=100, help='')

    parser.add_argument('--embedding_dim', type=int, default=128, help='the dimension of embedding features')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability of dropout layer')

    parser.add_argument('--hidden_size', type=int, default=64, help='')
    parser.add_argument('--temperature', type=int, default=None, help='')
    parser.add_argument('--controller_tanh_constant', type=int, default=None, help='')
    parser.add_argument('--controller_op_tanh_reduce', type=int, default=None, help='')

    parser.add_argument('--controller_lr', type=float, default=2e-4, help='')

    parser.add_argument('--r_lambda', type=float, default=2.5e-4, help='the coefficient lambda in reward function')

    parser.add_argument('--evaluate_controller_freq', type=int, default=500, help='evalation frequency for controller')

    parser.add_argument('-p', '--path', help='The path of imagenet', type=str, default='')
    parser.add_argument('-g', '--gpu', help='The gpu(s) to use', type=str, default='all')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')

    parser.add_argument('--distributed', default=True)
    parser.add_argument('-b', '--batch_size', help='The batch on every device for validation', type=int, default=256)
    parser.add_argument('-j', '--workers', help='Number of workers', type=int, default=2)

    parser.add_argument('-pre_arch', '--previous_arch', help='previous architecture', type=str, default="2,2,2,2,2:3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3:3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3")
    parser.add_argument('--supernet_path', help='path to super network', type=str, default="")
    parser.add_argument('--previous_controller', help='path to previous controller model', type=str, default="")
    parser.add_argument('--dataset_class_num', help='the number of classes of ImageNet subset', type=int, default=1000)
    parser.add_argument('--dataset_ratio', help='the ratio of subset/entire set', type=float, default=0.80)
    parser.add_argument('--dataset_split', help='the ratio for spliting a subset, for example 0.9 means: 0.9 for train and the rest 0.1 for validation', type=float, default=0.9)
    

    # parameters for train supernet
    parser.add_argument('--supernet_resume_train', default=False)
    parser.add_argument('--supernet_resume_path', help='The path of supernet trainings model state dict', type=str, default='')
    parser.add_argument('--num_batches_each_arch', help='the number of batches for training each subnetwork of supernet', type=int, default=5)
    parser.add_argument('--save_freq_of_supernet', help='every n epochs to save a supernet model dict', type=int, default=20)
    parser.add_argument('--supernet_epochs', help='epochs of training a supernet', type=int, default=100)

    # parameters for evaluation
    parser.add_argument('--eval_model', help='name of model to be evaluated (a string of subnetworks in mobilenet space)', type=str, default='adaxpert-1000')
    parser.add_argument('--pretrained_submodel_path', help='The path of the pretrained model', type=str, default='')

    args = parser.parse_args()

    return args

args = get_config()

if args.output is not None:
    os.makedirs(args.output, exist_ok=True)

