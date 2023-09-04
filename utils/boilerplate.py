"""
Module to contain non data-related boilerplate code
"""
import os
from typing import Union

import torch
import torch.distributed as dist


def gather_object(to_collect: any) -> list:
    """Gathers Python object to_collect and returns as list"""
    if dist.is_initialized():
        gather = [0 for _ in range(dist.get_world_size())]
        dist.all_gather_object(gather, to_collect)
    else:
        gather = [to_collect]

    return gather


def init_dist(rank: int, world_size: int, port: Union[str, int] = '12355'):
    """Initialises the distributed process group"""
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
