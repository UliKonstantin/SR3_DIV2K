import os
import random
import torch
from torch.utils.data import DataLoader
from DataLoader import *
# For image transformations
import torchvision.transforms as T
from Model.DiffusionBlock import *
from train import *
from utils import *
# Your custom modules
# from your_dataset_file import SRDataset  # or however you import SRDataset
# from your_split_file import split_data   # or however you import split_data
import json
import torch

# Import the data-preprocessing function
from data_preprocessing import preprocess_data

# Your model and training logic
from Model.DiffusionBlock import DiffusionModel
from train import train_ddpm
from Test import evaluate_checkpoints

def main(config_path="config.json"):
    # 1) Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2) Data Preprocessing (returns dataset and dataloader)
    train_sr_dataset, train_sr_loader = preprocess_data(config_path)

    # 3) Instantiate the Diffusion model
    ddpm = DiffusionModel(time_steps=config["time_steps"])

    # 4) Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 5) Rest of config extraction
    time_steps = config["time_steps"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    image_dims = (3, config["crop_size"][0], config["crop_size"][1])
    low_res_dims = (3, config["crop_size"][0] // config["scale_factor"], config["crop_size"][1] // config["scale_factor"])
    checkpoint_path = config["checkpoint_path"]
    if config["mode"]  == "Train":
        # 5) Train
        final_ckpt = train_ddpm(
            model=ddpm,
            time_steps=time_steps,
            epochs=epochs,
            batch_size= batch_size,
            device=device,
            image_dims=image_dims,
            low_res_dims=low_res_dims,  # example
            checkpoint_path=checkpoint_path,
            startpoint=0,
            dataAndLoader=(train_sr_dataset, train_sr_loader)
        )
        # 6) Save checkpoint (example name)
        torch.save(final_ckpt, "my_final_checkpoint.pt")
        print("Training complete. Checkpoint saved.")
    elif config["mode"]  == "Sample":
        opt = torch.optim.Adam(ddpm.model.parameters(), lr=1e-3)
        epoch = load_checkpoint(ddpm, opt, checkpoint_path, device=device)
        process_batch_and_save(train_sr_loader, ddpm, device, startpt = 0)

    elif config["mode"] == "Test":

        fid_device = device
        checkpoint_directory = "checkpoints/final checkpoint"
        evaluate_checkpoints(ddpm, config["valid_hr_dir"], checkpoint_directory, device=device, batch_size=16, fid_device=fid_device)

if __name__ == "__main__":
    main()