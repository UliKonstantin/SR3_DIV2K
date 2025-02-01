import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from Model.DiffusionBlock import DiffusionModel  # adjust import as needed
from DataLoader import SRDataset  # adjust import as needed

def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves a checkpoint dictionary containing:
      - model.state_dict()
      - optimizer.state_dict()
      - current epoch
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


import torch
from collections import OrderedDict


def add_model_prefix(state_dict):
    """Add 'model.' prefix to all keys in the state dict."""
    return OrderedDict({f"model.{k}": v for k, v in state_dict.items()})


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from all keys in the state dict."""
    return OrderedDict({k.replace("module.", "", 1): v for k, v in state_dict.items()})


def load_checkpoint(model, optimizer, checkpoint_path, device=None):
    """
    Tries to load a checkpoint in the following order:
    1. Load as is.
    2. If an error occurs, try adding 'model.' as a prefix to all keys.
    3. If an error occurs, try removing 'module.' as a prefix from all keys.
    4. If all fail, raise an error.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    try:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully (as is).")
    except RuntimeError as e1:
        print("Error loading checkpoint (as is). Trying with 'model.' prefix...")
        try:
            model.load_state_dict(add_model_prefix(state_dict))
            print("Checkpoint loaded successfully (after adding 'model.' prefix).")
        except RuntimeError as e2:
            print("Error loading checkpoint (with 'model.' prefix). Trying by removing 'module.' prefix...")
            try:
                model.load_state_dict(remove_module_prefix(state_dict))
                print("Checkpoint loaded successfully (after removing 'module.' prefix).")
            except RuntimeError as e3:
                print("Error loading checkpoint (after removing 'module.' prefix).")
                raise RuntimeError(
                    f"Failed to load checkpoint from {checkpoint_path}.\n\n"
                    f"Original error:\n{e1}\n\n"
                    f"Error after adding 'model.' prefix:\n{e2}\n\n"
                    f"Error after removing 'module.' prefix:\n{e3}"
                )

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {checkpoint_path}, resuming at epoch {epoch}")
    return epoch

def train_ddpm(model=None,
               time_steps=2000,
               epochs=20,
               batch_size=16,
               device="None",
               image_dims=(3, 128, 128),
               low_res_dims=(3, 32, 32),
               checkpoint_path=None,
               startpoint=0,
               dataAndLoader=None):
    """
    Trains the diffusion model for super-resolution.

    Args:
        model (nn.Module): The diffusion model to train. If None, a new DiffusionModel is created.
        time_steps (int): Number of diffusion time steps.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        device (str): Device to run training on ('cuda' or 'cpu' or 'mps').
        image_dims (tuple): Dimensions of high-resolution images (C, H, W).
        low_res_dims (tuple): Dimensions of low-resolution images (C, H, W).
        checkpoint_path (str, optional): Path to a pre-trained checkpoint.
                                         If provided, loads model & optimizer states before training.
        startpoint (int): Starting epoch index. Used if continuing training manually.
        dataAndLoader (tuple, optional): (Dataset, DataLoader). If None, a new one is created.

    Returns:
        dict: A checkpoint dictionary containing everything needed to save or reload the model, e.g.:
              {
                  "model_state_dict": ddpm.model.state_dict(),
                  "optimizer_state_dict": opt.state_dict(),
                  "epoch": current_epoch
              }
    """

    print(f"Using device: {device}")
    # 1) Initialize or use the provided model
    if model is None:
        ddpm = DiffusionModel(time_steps=time_steps)
    else:
        ddpm = model

    # 2) Setup dataset and dataloader
    c, hr_sz, _ = image_dims
    _, lr_sz, _ = low_res_dims

    if dataAndLoader is None:
        ds = SRDataset('DIV2K/DIV2K_train_HR/', hr_sz=hr_sz, lr_sz=lr_sz)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=2)
    else:
        ds, loader = dataAndLoader

    # 3) Set up optimizer and loss function
    opt = torch.optim.Adam(ddpm.model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="mean")

    # 4) Move model to device
    ddpm.model.to(device)

    # 5) Possibly resume from a checkpoint
    current_epoch = startpoint
    if checkpoint_path is not None:
        loaded_epoch = load_checkpoint(ddpm.model, opt, checkpoint_path, device)
        # If you want to keep using 'startpoint' as a manual offset, you can do:
        # current_epoch = max(current_epoch, loaded_epoch + 1)
        # but usually you just resume exactly:
        current_epoch = loaded_epoch + 1

    # 6) Training loop
    print()
    for ep in range(epochs):
        ddpm.model.train()
        print(f"Epoch {current_epoch}:")
        losses = []
        stime = time.time()

        for i, (x, y) in enumerate(loader):
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)

            ts = torch.randint(low=1, high=ddpm.time_steps, size=(bs,))
            gamma = ddpm.alpha_hats[ts].to(device)
            ts = ts.to(device)

            # Apply noise
            y_noisy, target_noise = ddpm.add_noise(y, ts)
            model_in = torch.cat([x, y_noisy], dim=1)

            # Predict noise
            predicted_noise = ddpm.model(model_in, gamma)
            loss = criterion(target_noise, predicted_noise)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if i % 250 == 0:
                print(f"  Step {i} | Loss: {loss.item():.6f} | Epoch {current_epoch}")
                checkpoint_name = f"sr_ep_{current_epoch}.pt"
                save_checkpoint(ddpm.model, opt, current_epoch, "/Users/UliKonstantin4/PycharmProjects/SR3/checkpoints/"+checkpoint_name)
        ftime = time.time()
        avg_loss = sum(losses) / len(losses)
        print(f"  Epoch trained in {ftime - stime:.2f}s; Avg loss => {avg_loss:.6f}")

        # 7) (Optional) Save a checkpoint each epoch
        #    E.g., if you want to automatically save progress:
        # checkpoint_name = f"sr_ep_{current_epoch}.pt"
        # save_checkpoint(ddpm.model, opt, current_epoch, checkpoint_name)

        current_epoch += 1
        print()

    # 8) Build a final checkpoint dictionary
    final_checkpoint = {
        "model_state_dict": ddpm.model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": current_epoch - 1
    }

    return final_checkpoint