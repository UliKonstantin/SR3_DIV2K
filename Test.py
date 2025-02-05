import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DataLoader import SRDataset
from train import load_checkpoint


# ----- Dummy Metric Functions -----
def compute_psnr(img1, img2):
    # Assumes images are in the range [0,1]
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def compute_ssim(img1, img2):
    # Replace with your actual SSIM implementation.
    return np.random.uniform(0.8, 1.0)


def compute_ms_ssim(img1, img2):
    # Replace with your actual multi-scale SSIM implementation.
    return np.random.uniform(0.8, 1.0)


def compute_fid(gen_images, real_images):
    # Replace with your FID implementation (e.g. torch-fidelity)
    return np.random.uniform(10, 50)


# ----- End Dummy Functions -----


# ----- Provided Sampling Function -----
def reconstruct(model, lr_img, device="cuda"):
    """
    Reconstructs a high-resolution image from a low-resolution image using the reverse diffusion process.

    Args:
        model: The diffusion model object. It is assumed that model has the following attributes:
               - time_steps: total number of diffusion steps.
               - alphas, alpha_hats, betas: diffusion schedule parameters.
               - The callable interface to predict noise: model(torch.cat([lr_img, y], dim=1), <gamma/tensor>)
        lr_img: Low-resolution input image tensor.
        device (str): Device for computation.

    Returns:
        y: The reconstructed high-resolution image tensor.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Start with random noise; shape should match lr_img.
        y = torch.randn_like(lr_img, device=device)
        lr_img = lr_img.to(device)

        # Reverse diffusion loop.
        for i, t in enumerate(range(model.time_steps - 1, 0, -1)):
            alpha_t = model.alphas[t]
            alpha_t_hat = model.alpha_hats[t]
            beta_t = model.betas[t]
            t_tensor = torch.tensor(t, device=device).long()

            # Predict noise using the current state.
            pred_noise = model(torch.cat([lr_img, y], dim=1), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1 / alpha_t)) * (y - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_hat)) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise

        return y


# ----- End Sampling Function -----


def evaluate_checkpoints(ddpm, val_dir, ckpt_dir, device="cuda", batch_size=16, num_workers=2, fid_device=None):
    """
    Loads checkpoints from a directory, runs the model on a validation set using the provided sampling routine,
    computes reconstruction loss (MSE), PSNR, SSIM, MS-SSIM, and FID, and plots the metrics versus epoch.

    Args:
        ddpm: The diffusion model object. It must contain the noise prediction network and diffusion parameters.
        val_dir (str): Directory containing validation images.
        ckpt_dir (str): Directory containing checkpoint files (e.g., "sr_ep_{epoch}.pt").
        device (str): Device for evaluation computations.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of DataLoader workers.
        fid_device (str, optional): Device on which to accumulate images for FID computation.
                                    Defaults to the same device as `device`.
    """
    # Set fid_device to evaluation device if not provided.
    if fid_device is None:
        fid_device = device

    # Create validation dataset and DataLoader.
    # (For evaluation, you might wish to disable randomness in the transforms.)
    val_dataset = SRDataset(val_dir, limit=-1, _transforms=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Find and sort checkpoint files (assuming names like "sr_ep_{epoch}.pt")
    ckpt_pattern = os.path.join(ckpt_dir, "sr_ep_*.pt")
    ckpt_files = sorted(glob.glob(ckpt_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Lists to store metrics per checkpoint.
    epochs_list = []
    loss_list = []
    psnr_list = []
    ssim_list = []
    mssim_list = []
    fid_list = []

    # Dummy optimizer (only needed for load_checkpoint).
    dummy_optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=1e-3)

    # Loop over each checkpoint.
    for ckpt_path in ckpt_files:
        print(f"Evaluating checkpoint: {ckpt_path}")
        epoch = load_checkpoint(ddpm.model, dummy_optimizer, ckpt_path, device)
        ddpm.model.to(device)
        ddpm.model.eval()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_mssim = 0.0
        total_samples = 0

        # For FID computation, accumulate generated and ground-truth images.
        gen_images_list = []
        gt_images_list = []

        with torch.no_grad():
            for x, y in val_loader:
                bs = y.shape[0]
                x, y = x.to(device), y.to(device)
                # Reconstruct HR image using the sample() function.
                sr = reconstruct(ddpm, x, device=device)
                loss = torch.nn.functional.mse_loss(sr, y)
                psnr = compute_psnr(sr, y)
                ssim = compute_ssim(sr, y)
                mssim = compute_ms_ssim(sr, y)

                total_loss += loss.item() * bs
                total_psnr += psnr * bs
                total_ssim += ssim * bs
                total_mssim += mssim * bs
                total_samples += bs

                # Accumulate images for FID computation on the specified fid_device.
                gen_images_list.append(sr.to(fid_device))
                gt_images_list.append(y.to(fid_device))

        avg_loss = total_loss / total_samples
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_mssim = total_mssim / total_samples

        gen_images_all = torch.cat(gen_images_list, dim=0)
        gt_images_all = torch.cat(gt_images_list, dim=0)
        fid_score = compute_fid(gen_images_all, gt_images_all)

        print(f"Epoch {epoch}: MSE Loss={avg_loss:.6f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, "
              f"MS-SSIM={avg_mssim:.4f}, FID={fid_score:.2f}")

        epochs_list.append(epoch)
        loss_list.append(avg_loss)
        psnr_list.append(avg_psnr)
        ssim_list.append(avg_ssim)
        mssim_list.append(avg_mssim)
        fid_list.append(fid_score)

    # Plot the metrics versus checkpoint epoch.
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(epochs_list, loss_list, marker='o', color='red')
    axs[0].set_title("Reconstruction Loss vs. Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE Loss")

    axs[1].plot(epochs_list, psnr_list, marker='o', label="PSNR", color='blue')
    axs[1].plot(epochs_list, ssim_list, marker='o', label="SSIM", color='green')
    axs[1].plot(epochs_list, mssim_list, marker='o', label="MS-SSIM", color='orange')
    axs[1].set_title("Image Quality Metrics vs. Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Metric Value")
    axs[1].legend()

    axs[2].plot(epochs_list, fid_list, marker='o', color='purple')
    axs[2].set_title("FID vs. Epoch")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("FID Score")

    plt.tight_layout()
    plt.show()

