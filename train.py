import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from utils.transforms import RandomRotFlip, RandomCrop, RandomBrightnessContrast, RandomGaussianNoise, ToTensor

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler


from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/opt/data/private/data_3D',
                    help='Name of Experiment')

# parser.add_argument('--dataset', type=str, default='/2018LA_Seg_Training Set',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/Pancreas',
#                     help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='/BraTS2019',
                    help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/Lung',
#                     help='Name of Experiment')


parser.add_argument('--labeled_num', type=int, default=10,
                    help='Percentage of label quantity')

parser.add_argument('--nms', type=int,  default=True,
                    help='output channel of network')

parser.add_argument('--in_channels', type=int,  default=1,
                    help='output channel of network')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')

# parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
#                     help='patch size of network input')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')

parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1, help='maximum epoch number to train')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float, default=0.9, help='ema_decay')

parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    patch_size = args.patch_size

    # model
    trainer = Trainer(args)

    # dataset
    if args.dataset == "/2018LA_Seg_Training Set":
        train_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split="train_LA",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size),  ToTensor()]))
    elif args.dataset == "/Pancreas":
        train_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split="train_Pancreas",
                                      transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]))
    elif args.dataset == "/BraTS2019":
        train_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split="train_BraTS2019",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
    elif args.dataset == "/Lung":
        train_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split="train_Lung",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
    
    # sampler
    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    logging.info("{} iterations per epoch".format(len(train_loader)))
    # max_epoch = max_iterations // len(train_loader) + 1
    max_epoch = (max_iterations) // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = 0

    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            trainer.train(sampled_batch, iter_num, snapshot_path)
            iter_num = iter_num + 1
            if iter_num > 0 and iter_num % 200 == 0:
                trainer.test(snapshot_path, iter_num)


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        random.seed(fold*5 + 42)
        np.random.seed(fold*10 + 42)
        torch.manual_seed(fold*100 + 42)
        torch.cuda.manual_seed(fold*1000 + 42)

        snapshot_path = "./Results/result_BraTS_10/fold_" + str(fold)

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train.py", snapshot_path + "/code/train.py")
        shutil.copyfile("./trainer.py", snapshot_path + "/code/trainer.py")
        shutil.copyfile("./model/Semi_MoE.py", snapshot_path + "/code/Semi_MoE.py")

        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        train(args, snapshot_path)