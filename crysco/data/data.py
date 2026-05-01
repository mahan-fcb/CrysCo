"""
Data Loading and Dataset Management for CrysCo

This module handles dataset loading, data splitting, and PyTorch Geometric
DataLoader setup for the CrysCo model training pipeline.
"""

import os
import numpy as np
import torch

from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from ..utils.utils_train import split_data
def setup_data_loaders(
    train_ratio,
    validation_ratio,
    test_ratio,
    batch_size,
    dataset,
    device,
    random_seed,
    world_size=0,
    num_workers=0,
):
    """
    Set up train, validation, and test data loaders with optional distributed training support.

    Args:
        train_ratio: Fraction of data for training
        validation_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        batch_size: Batch size for data loaders
        dataset: PyTorch Geometric dataset
        device: Device for training ('cpu', 'cuda', or device for distributed)
        random_seed: Random seed for reproducible data splits
        world_size: Number of processes for distributed training
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_sampler, train_dataset, val_dataset, test_dataset)
    """
    # Split datasets
    train_dataset, validation_dataset, test_dataset = split_data(
        dataset, train_ratio, validation_ratio, test_ratio, random_seed
    )

    # Setup distributed sampling if needed
    if device not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, device=device
        )
    else:
        train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    if device in (0, "cpu", "cuda"):
        if len(validation_dataset) > 0:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        validation_dataset,
        test_dataset,
    )



def get_dataset(data_path, filename="Eh.pt", target_index=0):
    processed_path = "processed"
    transforms = GetY(index=target_index)

    if os.path.exists(os.path.join(data_path, processed_path, filename)):
        dataset = StructureDataset(
            data_path,
            processed_path,
            filename=filename,
            transform=transforms,
        )
        return dataset


##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", filename="Eh.pt", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        self.filename = filename  # Add this line
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        # Handle PyTorch 2.6+ weights_only default change
        try:
            self.data, self.slices = torch.load(
                os.path.join(self.processed_dir, self.filename),
                weights_only=False
            )
        except Exception as e:
            # Fallback for older PyTorch versions
            self.data, self.slices = torch.load(os.path.join(self.processed_dir, self.filename))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        return [self.filename]
##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data



