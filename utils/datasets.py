"""
Standardised data loading
"""
from typing import Union, Type

import torch

from torch.utils.data import Dataset, Subset, DistributedSampler, DataLoader
from torchvision.datasets import OxfordIIITPet, CIFAR10, CIFAR100, VisionDataset
from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, Resize, CenterCrop
from torchvision.transforms.functional import InterpolationMode
from timm.data.transforms import RandomResizedCropAndInterpolation

from .data import DogDataset

DATASETS = {
    'oxford-iiit': 37,
    'cifar-10': 10,
    'cifar-100': 100,
    'dogs': 269
}
DATA_PATH = 'data'


def get_transforms_train(size: int = 224) -> Compose:
    """Standard training transforms used by ResNet"""
    return Compose([
        RandomResizedCropAndInterpolation(size=(size, size), interpolation='bicubic'),
        RandomHorizontalFlip(),
        ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_transforms_test(resize: int = 256, size: int = 224) -> Compose:
    """Standard testing transforms used by ResNet"""
    return Compose([
        Resize(size=resize, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size=(size, size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_num_classes(dataset: str) -> int:
    """Returns the number of classes in a given dataset"""
    return DATASETS[dataset]


class DataContainer:
    """Convenience class for passing dataloaders/samplers to functions"""
    def __init__(self, rank: int, world_size: int, batch_size: int, train_dataset: Dataset = None,
                 valid_dataset: Dataset = None, test_dataset: Dataset = None):
        train_dataloader, valid_dataloader, test_dataloader = None, None, None
        train_sampler, valid_sampler, test_sampler = None, None, None

        if train_dataset:
            train_dataloader, train_sampler = get_loader(train_dataset, world_size, rank, batch_size)
        if valid_dataset:
            valid_dataloader, valid_sampler = get_loader(valid_dataset, world_size, rank, batch_size)
        if test_dataset:
            test_dataloader, test_sampler = get_loader(test_dataset, world_size, rank, batch_size)

        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler

        self.valid_dataloader = valid_dataloader
        self.valid_sampler = valid_sampler

        self.test_dataloader = test_dataloader
        self.test_sampler = test_sampler


def _get_idxs(dataset: Type[VisionDataset], split: float = 0.9) -> (torch.Tensor, torch.Tensor):
    """Returns the indices for a train/valid split of the training data"""
    len_dataset = len(dataset(DATA_PATH, download=True))
    lim = int(split * len_dataset)
    idxs = torch.randperm(len_dataset)
    train_idxs, valid_idxs = idxs[:lim], idxs[lim:]
    return train_idxs, valid_idxs


def get_dataset(name: str, train=True, valid=True, test=False, transforms_train=None, transforms_test=None) -> tuple:
    """Loads all required datasets in a standardised way"""
    if name not in DATASETS.keys():
        raise ValueError(f'Dataset "{name}" is not recognised')

    # Check that the required transforms are provided
    if train:
        assert transforms_train
    if valid or test:
        assert transforms_test

    train_dataset, valid_dataset, test_dataset = None, None, None

    # Load requested dataset
    if name == 'oxford-iiit':
        if train or valid:
            # Need to split the 'testval' set into a test and validation split
            train_idxs, valid_idxs = _get_idxs(OxfordIIITPet)

        if train:
            train_dataset = Subset(
                OxfordIIITPet(DATA_PATH, transform=transforms_train),
                train_idxs
            )
        if valid:
            valid_dataset = Subset(
                OxfordIIITPet(DATA_PATH, transform=transforms_test),
                valid_idxs
            )
        if test:
            test_dataset = OxfordIIITPet(DATA_PATH, split='test', transform=transforms_test)
    elif name == 'cifar-10':
        if train or valid:
            # Need to split the 'testval' set into a test and validation split
            train_idxs, valid_idxs = _get_idxs(CIFAR10)

        if train:
            train_dataset = Subset(
                CIFAR10(DATA_PATH, transform=transforms_train),
                train_idxs
            )
        if valid:
            valid_dataset = Subset(
                CIFAR10(DATA_PATH, transform=transforms_test),
                valid_idxs
            )
        if test:
            test_dataset = CIFAR10(DATA_PATH, train=False, transform=transforms_test)
    elif name == 'cifar-100':
        if train or valid:
            # Need to split the 'testval' set into a test and validation split
            train_idxs, valid_idxs = _get_idxs(CIFAR100)

        if train:
            train_dataset = Subset(
                CIFAR100(DATA_PATH, transform=transforms_train),
                train_idxs
            )
        if valid:
            valid_dataset = Subset(
                CIFAR100(DATA_PATH, transform=transforms_test),
                valid_idxs
            )
        if test:
            test_dataset = CIFAR100(DATA_PATH, train=False, transform=transforms_test)
    elif name == 'dogs':
        if train:
            train_dataset = DogDataset(DATA_PATH, split='train', transform=transforms_train)
        if valid:
            valid_dataset = DogDataset(DATA_PATH, split='valid', transform=transforms_test)
        if test:
            test_dataset = DogDataset(DATA_PATH, split='test', transform=transforms_test)

    return train_dataset, valid_dataset, test_dataset


def get_loader(dataset: Dataset, world_size: int, rank: int, batch_size: int, drop_last: bool = False)\
        -> (DataLoader, Union[DistributedSampler, None]):
    """Creates a data loader and (optionally) a distributed sampler if world_size > 1"""
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last
    )

    return dataloader, sampler
