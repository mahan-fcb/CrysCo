"""
Training Utilities for CrysCo Model

This module provides training loop implementations, model setup utilities,
evaluation functions, and distributed training support for the CrysCo
hybrid graph-transformer model.
"""

import csv
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from ..models.CrysCo import CrysCo


def train_one_epoch(model, optimizer, data_loader, loss_function_name, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        predictions = model(batch_data)
        loss_function = getattr(F, loss_function_name)
        loss = loss_function(predictions, batch_data.y)

        loss.backward()
        total_loss += loss.detach() * predictions.size(0)
        optimizer.step()

        total_samples += predictions.size(0)

    average_loss = total_loss / total_samples
    return average_loss




def train_model(
    device,
    world_size,
    model,
    optimizer,
    scheduler,
    loss_function_name,
    train_loader,
    validation_loader,
    train_sampler,
    num_epochs,
    verbosity,
    model_save_path="my_model_temp.pth",
):
    """Main training loop with validation and model checkpointing."""

    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    best_state_dict = model.state_dict()  # Initialize with the initial model state

    # Start training over epochs loop
    for epoch in range(1, num_epochs + 1):

        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        if device not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)

        # Train model for one epoch
        train_error = train_one_epoch(model, optimizer, train_loader, loss_function_name, device)
        if device not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        # Get validation performance
        if device not in ("cpu", "cuda"):
            dist.barrier()
        if validation_loader is not None and device in (0, "cpu", "cuda"):
            val_error = evaluate(validation_loader, model, loss_function_name, device, out=False)

        # Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        # Remember the best validation error and save model checkpoint
        if validation_loader is not None and device in (0, "cpu", "cuda"):
            if val_error < best_val_error:
                best_state_dict = model.state_dict()
                best_val_error = val_error
                torch.save(
                    {
                        "state_dict": best_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    model_save_path,
                )

        # Update the learning rate scheduler
        if validation_loader is not None and device in (0, "cpu", "cuda"):
            scheduler.step(val_error)
        else:
            scheduler.step(train_error)

        # Print performance
        if epoch % verbosity == 0 and device in (0, "cpu", "cuda"):
            print(
                f"Epoch: {epoch:04d}, Learning Rate: {current_lr:.6f}, "
                f"Training Error: {train_error:.5f}, Val Error: {val_error:.5f}, "
                f"Time per epoch (s): {epoch_time:.5f}"
            )

    if device not in ("cpu", "cuda"):
        dist.barrier()

    # Load the best model state to return
    model.load_state_dict(best_state_dict)
    return model


def model_setup(
    rank,
    model_name,
    model_params,
    dataset,
    load_model=False,
    model_path=None,
    print_model=True,
):
    model = CrysCo(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(rank)
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])

    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
    return model



def split_data(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = int(dataset_size * test_ratio)
        unused_size = dataset_size - train_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size, unused_size],
            generator=torch.Generator().manual_seed(seed),
        )
        
        # Determine the sizes of validation and test data to be added to the training set
        tv= int(val_size * 0.41)
        ts = int(test_size * 0.41)
        
        # Subset validation and test datasets
        train = torch.utils.data.Subset(val_dataset, range(tv))
        ttrain = torch.utils.data.Subset(test_dataset, range(ts))
        
        # Concatenate subsets to the training dataset
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train, ttrain])
        
        print(
            "train length:", len(train_dataset),
            "val length:", len(val_dataset) ,
            "test length:", len(test_dataset) ,
            "unused length:", unused_size,
            "seed:", seed,
        )
        
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")

def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.size(0)
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
                else:
                    ids_temp = [
                        item for sublist in data.structure_id for item in sublist
                    ]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids = ids + ids_temp
                    predict = np.concatenate(
                        (predict, output.data.cpu().numpy()), axis=0
                    )
                    target = np.concatenate((target, data.y.cpu().numpy()), axis=0)
            count = count + output.size(0)

    loss_all = loss_all / count

    if out == True:
        test_out = np.column_stack((ids, target, predict))
        return loss_all, test_out
    elif out == False:
        return loss_all


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])
