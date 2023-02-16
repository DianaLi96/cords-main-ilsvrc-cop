import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn


def init_dist(backend='nccl', master_ip='tcp://127.0.0.1', port=6669):
    print('Start!', '++++++++++++++++++++++')

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_ADDR'] = master_ip
    # os.environ['MASTER_PORT'] = str(port)
    
    print('we are here now', '===========')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    # dist_url = master_ip + ':' + str(port)
    # dist.init_process_group(backend=backend, init_method=dist_url, world_size=world_size, rank=rank)
    dist.init_process_group(backend=backend)
    return rank, world_size


def average_gradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)