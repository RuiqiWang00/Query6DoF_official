import torch
import numpy as np
from .pose_dataset import DATASET_REGISTRY
from mmengine import Registry
from torch.utils.data import DataLoader
import copy

def build_unsupervise_dataloader(cfg, registry, *args, **kwargs):
    dataset=DATASET_REGISTRY.build(cfg.REAL_DATASET)
    loader_cfg=copy.deepcopy(cfg)
    loader_cfg=cfg.REAL_DATALOADER
    loader_cfg.dataset=dataset
    if 'TRAIN' in cfg.keys():
        if cfg.DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=None, generator=None)
        
        loader_cfg.sampler=train_sampler
    data_loader = registry.get(cfg.REAL_DATALOADER.type)
    del loader_cfg['type']
    data_loader=data_loader(**loader_cfg)

    dataset=DATASET_REGISTRY.build(cfg.CAMERA_DATASET)
    loader_cfg=copy.deepcopy(cfg)
    loader_cfg=cfg.CAMERA_DATALOADER
    loader_cfg.dataset=dataset
    if 'TRAIN' in cfg.keys():
        if cfg.DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler =  torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=None, generator=None)
        
        
        loader_cfg.sampler=train_sampler
    data_loader2 = registry.get(cfg.CAMERA_DATALOADER.type)
    del loader_cfg['type']
    data_loader2=data_loader2(**loader_cfg)

    return (data_loader, data_loader2)

UNSUPERVISE_DATALOADER_REGISTRY = Registry("UNSUPERVISE_DATALOADER_REGISTRY",build_func=build_unsupervise_dataloader)
UNSUPERVISE_DATALOADER_REGISTRY.register_module(module=DataLoader)

def build_dataloader(cfg, registry, *args, **kwargs):
    dataset=DATASET_REGISTRY.build(cfg.DATASET)
    loader_cfg=copy.deepcopy(cfg)
    loader_cfg=cfg.DATALOADER
    loader_cfg.dataset=dataset
    if 'TRAIN' in cfg.keys():
        if cfg.DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=None, generator=None)
        
        loader_cfg.sampler=train_sampler
    data_loader = registry.get(cfg.DATALOADER.type)
    del loader_cfg['type']
    data_loader=data_loader(**loader_cfg)
    return data_loader

DATALOADER_REGISTRY = Registry("DATALODER",build_func=build_dataloader)
DATALOADER_REGISTRY.register_module(module=DataLoader)

def trivial_batch_collator(batch):
    return batch


