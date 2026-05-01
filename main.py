
"""
CrysCo Training Script

This script handles the training of the CrysCo hybrid graph-transformer model
for materials property prediction. It includes model configuration, training
parameters, and the main training loop with distributed training support.

Usage:
    python main.py --data_dir <project_directory> --data <dataset_file>
"""

import argparse
import csv
import os
import sys
import time
import json
import warnings
import shutil
import copy
import glob
from datetime import datetime
from functools import partial
import platform

import numpy as np
import ase
from ase import io
from scipy.stats import rankdata
from scipy import interpolate

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from crysco.models.CrysCo import CrysCo
from crysco.utils.utils_train import train_model, train_one_epoch, model_setup, model_summary, evaluate, write_results
from crysco.data.data import setup_data_loaders, get_dataset, StructureDataset, GetY

model_parameters = {  
       "out_dims":64,
        "d_model":128,
        "N":3,
        "heads":4
        ,"dim1": 64
        ,"dim2": 64
        ,"numb_embbeding":1
        ,"numb_EGAT":5
        ,"numb_GATGCN":1
        ,"pool": "global_add_pool"
        ,"pool_order": "early"
        ,"act": "silu"
        ,"model": "CrysCo"
        ,"dropout_rate": 0.0
        ,"epochs": 100
        ,"lr": 0.006
        ,"batch_size": 80
        ,"optimizer": "AdamW"
        ,"optimizer_args": {}
        ,"scheduler": "ReduceLROnPlateau"
        ,"scheduler_args": {"mode":"min", "factor":0.8, "patience":15, "min_lr":0.00001, "threshold":0.0002}}
training_parameters = { 
    "target_index": 0
    ,"loss": "mse_loss"       
    ,"verbosity": 1
}
job_parameters= { 
        "reprocess":"False"
        ,"job_name": "my_train_job"   
        ,"load_model": "False"
        ,"save_model": "True"
        ,"model_path": "my_model_shear2.pth"
        ,"write_output": "True"
        ,"parallel": "True"
}
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the CrysCo model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--data", type=str, required=True, help="Dataset file name (.pt)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use: cuda or cpu")
    parser.add_argument("--epochs", type=int, default=model_parameters["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=model_parameters["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=model_parameters["lr"], help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Test data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_path", type=str, default=job_parameters["model_path"], help="Path to save model")
    parser.add_argument("--job_name", type=str, default=job_parameters["job_name"], help="Job name for output files")
    args = parser.parse_args()

    # Override model_parameters with CLI args
    model_parameters["epochs"] = args.epochs
    model_parameters["batch_size"] = args.batch_size
    model_parameters["lr"] = args.lr

    # Check CUDA availability and fall back to CPU if needed
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")
    print(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Data split: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")

    # Setup the dataset
    database = get_dataset(args.data_dir, args.data, 0)

    # Setup loaders
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = setup_data_loaders(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.batch_size,
        database,
        device,
        args.seed,
        0,
    )

    # Setup the model
    model = model_setup(
        device,
        'CrysCo',
        model_parameters,
        database,
    )

    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    train_model(
            device,
            0,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            val_loader,
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            "my_model_temp.pth")


    train_error = val_error = test_error = float("NaN")

    ##workaround to get training output in DDP mode
    ##outputs are slightly different, could be due to dropout or batchnorm?
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_parameters["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Get train error in eval mode
    train_error, train_out = evaluate(
        train_loader, model, training_parameters["loss"], device, out=True
    )
    print("Train Error: {:.5f}".format(train_error))

    ##Get val error
    if val_loader != None:
        val_error, val_out = evaluate(
            val_loader, model, training_parameters["loss"], device, out=True
        )
        print("Val Error: {:.5f}".format(val_error))

    ##Get test error
    if test_loader != None:
        test_error, test_out = evaluate(
            test_loader, model, training_parameters["loss"], device, out=True
        )
        print("Test Error: {:.5f}".format(test_error))


    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "full_model": model,
        },
        args.model_path,
    )

    ##Write outputs
    if job_parameters["write_output"] == "True":

        write_results(
            train_out, str(args.job_name) + "_train_outputs.csv"
        )
        if val_loader != None:
            write_results(
                val_out, str(args.job_name) + "_val_outputs.csv"
            )
        if test_loader != None:
            write_results(
                test_out, str(args.job_name) + "_test_outputs.csv"
            )


if __name__ == "__main__":
    main()
