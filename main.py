import torch.nn.functional as F
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
from dataset.dataset import *
from model.model import *
from loss import *
from utils import *
from train_pretext import *
from optimizer import *
import logging
import json

rand_seed = 42

def process(args):
    path = args.path
    data_name = args.data_name
    data_path = path + f'data/{data_name}/'
    data_df = pd.read_csv(path + f'data/{data_name}.csv')
    device = args.device
    gpu_id = args.gpu_id

    datassl = DatasetSSL(data_path=data_path, df=data_df)
    datads = DatasetDownstream(data_path=data_path, df=data_df)

    model = Model(Nh=224, Nw=224, bs=32, ptsz = 32, pout = 512, num_tokens=169, dim_in=512,  dim=512, heads = 8, dim_head = 64, dropout = 0.).to(device)

    checkpoints_path = f'checkpoints/' + data_name + '/'
    if not os.path.exists(checkpoints_path):
      os.makedirs(checkpoints_path)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    pretext(args, datassl, model.backbone)
    downstream(args, datads, model)


def pretext(args, dataset, model):
    train_indices, val_indices = split_indices(len(dataset), args.pre_val_pct, rand_seed)
    batch_size = args.pre_bs
    # Training sampler and data loader
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(dataset, batch_size, sampler=train_sampler, num_workers=2)

    # Validation set and data loader
    val_sampler = SubsetRandomSampler(val_indices)
    val_dl = DataLoader(dataset, batch_size, sampler=val_sampler)

    train_dl = DeviceDataLoader(train_dl, args.device)
    val_dl = DeviceDataLoader(val_dl, args.device)

    # loss_func = CorLoss(batch_size=batch_size)
    loss_func = SSLLoss()

    # opt = LARS
    opt = torch.optim.Adam
    lr = args.pre_lr
    epochs = args.pre_eps

    train_losses, val_losses, _ = train_pretext(epochs, model, loss_func, train_dl, val_dl, opt_fn=opt, lr=lr, metric=None, expt_name=args.expt_name, PATH=args.path)



def downstream(args, dataset, model):
    train_indices, val_indices = split_indices(len(dataset), args.val_pct, rand_seed)
    batch_size = args.batch_size
    # Training sampler and data loader
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)

    # Validation set and data loader
    val_sampler = SubsetRandomSampler(val_indices)
    val_dl = DataLoader(datassl, batch_size, sampler=val_sampler)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_name', type=str, default='cedar')
    parser.add_argument('--path', type=str, default= "C:\\Users\\RedmiBook\\HUST\\Documents\\Studying\\Phase_2_VDT\\code\\Project_code\\")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--pre_bs', type=int, default=16)
    parser.add_argument('--pre_lr', type=float, default=1e-3)
    parser.add_argument('--pre_eps', type=int, default=20)
    parser.add_argument('--pre_val_pct', type=float, default=0.1)

    parser.add_argument('--ds_bs', type=int, default=16)
    parser.add_argument('--ds_lr', type=float, default=1e-3)
    parser.add_argument('--ds_eps', type=int, default=20)

    parser.add_argument('--expt_name', type=str, default='cedar')
    args = parser.parse_args()
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT)
    d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
#   logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    process(args)