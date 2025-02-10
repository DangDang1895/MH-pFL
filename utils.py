import torch
import random
import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args(parser):
    parser.add_argument("--norm-var", type=float, default=0.002, help="")
    parser.add_argument("--embed-dim", type=int, default=64, help="embedding dim")
    parser.add_argument("--grad-clip", type=int, default=50, help="")
    parser.add_argument("--random-seed", type=int, default=42, help="")
    parser.add_argument("--data-name", type=str, default="tiny_imageNet", choices=['cifar10', 'cifar100', 'tiny_imageNet','emnist'], help="dir path for dataset")
    parser.add_argument("--data-path", type=str, default="data", help="dir path for data")
    parser.add_argument("--data-distribution", type=str, default="incomplete_label", choices=['dirichlet', 'incomplete_label'])
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'adamw'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")

    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--inner-wd", type=float, default=5e-4, help="inner weight decay")
    parser.add_argument("--cuda", type=int, default=0, help="gpu device ID")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument("--least-nums", type=int, default=30, help="")
    parser.add_argument("--hnet-output-size", type=int, default=3072, help="")
    parser.add_argument("--hidm", type=int, default=128, help="")
    parser.add_argument("--hidden-layers", type=int, default=3, help="")
    parser.add_argument("--save-dir", type=str, default="Output", help="")
    parser.add_argument("--topk", type=str2bool, default=False, help="")
    parser.add_argument("--only-cnn", type=str2bool, default=False, help="Only CNN model") 

    parser.add_argument("--train-clients", type=int, default=-1, help="train first # clients")
    parser.add_argument("--test-clients", type=int, default=-1, help="")
    parser.add_argument("--test-more-model", type=str2bool, default=False, help="")
    parser.add_argument("--save-model", type=str2bool, default=False, help="")
    parser.add_argument("--hynet-dir", type=str, default="", help="")
    parser.add_argument("--fc-dir", type=str, default="", help="")
    parser.add_argument("--alpha", type=float, default=0.2, help="0.01,0.05,0.1,0.15,0.2")
    parser.add_argument("--temp", type=int, default=15, help="5,10,15,20,25")

    args = parser.parse_args()
    return args
