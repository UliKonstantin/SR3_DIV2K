import torchshow as ts
import torchvision.utils as vutils

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
import os

def sample(model, lr_img, device="cuda"):
    """
    Runs the reverse diffusion process on a batch of LR images and returns
    the reconstructed HR images.

    Args:
        model: Your diffusion model (which has attributes: time_steps, alphas, alpha_hats, betas).
        lr_img: Batch of low-resolution images.
        device (str): Device to run computations on.

    Returns:
        y: Batch of reconstructed images.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Initialize with random noise; shape matches lr_img.
        y = torch.randn_like(lr_img, device=device)
        lr_img = lr_img.to(device)

        for i, t in enumerate(range(model.time_steps - 1, 0, -1)):
            alpha_t = model.alphas[t]
            alpha_t_hat = model.alpha_hats[t]
            beta_t = model.betas[t]
            t_tensor = torch.tensor(t, device=device).long()
            # Concatenate lr_img and current state y, then predict noise.
            pred_noise = model(torch.cat([lr_img, y], dim=1), alpha_t_hat.view(-1).to(device))
            # Reverse diffusion update:
            y = (torch.sqrt(1 / alpha_t)) * (y - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_hat)) * pred_noise)
            if t_tensor > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise
        return y


def process_batch_and_save(loader, model, device="cuda", save_dir="figures",startpt = 0):
    """
    Processes one batch from the loader: obtains LR, HR, and reconstructed images,
    then for each sample in the batch creates a side-by-side plot with:
      - Left: Low-Resolution image.
      - Center: High-Resolution (ground truth) image.
      - Right: Reconstructed image.
    Each plot is saved to the specified directory.

    Args:
        loader: DataLoader yielding (LR, HR) image pairs.
        model: Diffusion model used for reconstruction.
        device (str): Device for computations.
        save_dir (str): Directory path where the figures will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get one batch (here we process only the first batch)
    for lr_img, hr_img in loader:
        # Optionally, print shapes for debugging.
        print(f"LR shape: {lr_img.shape}, HR shape: {hr_img.shape}")
        break

    lr_img = lr_img.to(device)
    hr_img = hr_img.to(device)

    # Get the reconstructed images by sampling.
    rec_img = sample(model, lr_img, device=device)

    # Determine the batch size.
    batch_size = lr_img.size(0)
    idx = startpt
    # For each image in the batch, plot side-by-side and save.
    for i in range(batch_size):
        idx+=1
        # Convert tensors to numpy arrays (HWC format) and clip to [0,1]
        lr_np = lr_img[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        hr_np = hr_img[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        rec_np = rec_img[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

        # Create a figure with 1 row and 3 columns.
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Left: LR image.
        axs[0].imshow(lr_np)
        axs[0].set_title("Low Resolution")
        axs[0].axis("off")

        # Center: HR image.
        axs[1].imshow(hr_np)
        axs[1].set_title("High Resolution (Ground Truth)")
        axs[1].axis("off")

        # Right: Reconstructed image.
        axs[2].imshow(rec_np)
        axs[2].set_title("Reconstructed")
        axs[2].axis("off")

        # Optional: overall title for the sample.
        fig.suptitle(f"Sample {idx}", fontsize=16)

        # Save the figure to the provided directory.
        save_path = os.path.join(save_dir, f"sample_{idx:03d}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved sample {idx} to {save_path}")
