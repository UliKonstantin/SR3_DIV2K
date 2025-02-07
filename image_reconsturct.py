import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import sample
from train import load_checkpoint
#--------------------------------------------------------------#
# Given a directory with image patches and each with name format <image-id_xcord_ycord> this function recomputes the hr lr and sr
#full image and compares them.
#--------------------------------------------------------------#
def process_hr_patches_and_save_full_image(patch_dir, model, checkpoint_path = None,optimizer = None,device="cuda", output_path="full_image.png",
                                           image_id=None, hr_size=128, lr_size=32):
    """
    Loads high-resolution patch images from a directory (with filenames like "0052_128_256.png"),
    computes a low-resolution version (by downscaling then upscaling), passes the LR patches
    through the super-resolution model, and then reconstructs full images by stitching the patches
    at their proper positions. Finally, it creates a side-by-side plot of:
        - Left: Full low-resolution (upscaled) image.
        - Right: Full super-resolved image.

    Args:
        patch_dir (str): Directory containing the high-resolution patch image files.
        model: Your super-resolution model.
        device (str): Device to run the model on.
        output_path (str): File path for saving the final figure.
        image_id (str or None): If provided, only process patches whose filename starts with this image_id (e.g. "0052").
        hr_size (int): The expected size (width/height) of each high-resolution patch (default: 128).
        lr_size (int): The size to which the HR patch is downscaled before being upscaled (default: 32).

    Note:
        This function assumes that there is a callable `sample(model, lr_tensor, device=device)`
        that takes a batch of LR patches (tensor shape: [N, C, hr_size, hr_size]) and returns the
        corresponding super-resolved patches.
    """

    # List patch files in the directory
    patch_files = [f for f in os.listdir(patch_dir) if os.path.isfile(os.path.join(patch_dir, f))]
    if image_id is not None:
        patch_files = [f for f in patch_files if f.startswith(image_id + "_")]

    if not patch_files:
        print("No patch files found!")
        return

    # We'll use a simple ToTensor transform (values in [0,1]) for later plotting.
    to_tensor = transforms.ToTensor()
    if checkpoint_path is not None:
        dummy_optimizer = optimizer
        epoch = load_checkpoint(model.model, dummy_optimizer, checkpoint_path, device)
        model.model.to(device)
        model.model.eval()
    # Store each patch along with its coordinates.
    patches_info = []
    for fname in patch_files:
        # Expected filename format: "0052_<x>_<y>.png"
        name, _ = os.path.splitext(fname)
        parts = name.split('_')
        if len(parts) < 3:
            print(f"Skipping {fname}: does not follow the naming convention.")
            continue

        try:
            # Here, parts[1] is the x coordinate and parts[2] is the y coordinate (both multiples of hr_size)
            x = int(parts[1])
            y = int(parts[2])
        except ValueError:
            print(f"Skipping {fname}: cannot parse coordinates.")
            continue

        patch_path = os.path.join(patch_dir, fname)
        # Load the HR patch (as a PIL image) and ensure it is in RGB.
        hr_patch_img = Image.open(patch_path).convert("RGB")
        # (Optional) Force the patch to be hr_size x hr_size if needed.
        if hr_patch_img.size != (hr_size, hr_size):
            hr_patch_img = hr_patch_img.resize((hr_size, hr_size), resample=Image.BICUBIC)

        # Compute the LR version:
        # 1. Downscale to lr_size x lr_size using bicubic interpolation.
        lr_patch_img = hr_patch_img.resize((lr_size, lr_size), resample=Image.BICUBIC)
        # 2. Upscale back to hr_size x hr_size using bicubic interpolation.
        lr_patch_img = lr_patch_img.resize((hr_size, hr_size), resample=Image.BICUBIC)

        # Convert both images to tensors.
        # (The HR patch is the ground truth if needed; here we mainly use the LR version.)
        hr_tensor = to_tensor(hr_patch_img)
        lr_tensor = to_tensor(lr_patch_img)

        patches_info.append({
            "coords": (x, y),
            "lr_tensor": lr_tensor,
            "hr_tensor": hr_tensor  # stored in case you want to compare with ground truth
        })

    if not patches_info:
        print("No valid patch information found!")
        return

    # Determine the full image dimensions.
    # Assumes that x and y in the filenames represent the top-left pixel coordinates.
    patch_dim = hr_size  # each patch is hr_size x hr_size
    max_x = max(info["coords"][0] for info in patches_info)
    max_y = max(info["coords"][1] for info in patches_info)
    full_width = max_x + patch_dim
    full_height = max_y + patch_dim

    # Assume all images have the same number of channels (e.g., 3 for RGB)
    C = patches_info[0]["lr_tensor"].shape[0]
    # Create empty canvases for the full LR (upscaled) and full super-resolved images.
    full_lr = torch.zeros((C, full_height, full_width))
    full_sr = torch.zeros((C, full_height, full_width))

    # Prepare a list of LR patches and corresponding coordinates.
    lr_list = [info["lr_tensor"] for info in patches_info]
    coords_list = [info["coords"] for info in patches_info]

    # Stack the LR patches into a batch and move to the specified device.
    lr_batch = torch.stack(lr_list, dim=0).to(device)

    # Process the LR patches through your super-resolution model.
    # (Assumes your sample() function handles batched input.)
    with torch.no_grad():

        sr_batch = sample(model, lr_batch, device=device)

    sr_batch = sr_batch.detach().cpu()

    # Place each patch (both LR and SR) into its correct location in the full image.
    for i, (x, y) in enumerate(coords_list):
        save_path = os.path.join("reconstruct_lr_patches", f"sample_{i:03d}.png")
        plt.figure(figsize=(8, 8))
        plt.title("Low Resolution Patch")
        plt.imshow(lr_batch[i].detach().cpu().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(save_path)
        if i<2:
            plt.show()

        full_lr[:, y:y + patch_dim, x:x + patch_dim] = lr_batch[i].detach().cpu()
        full_sr[:, y:y + patch_dim, x:x + patch_dim] = sr_batch[i]

    # Convert the full images to numpy arrays for plotting (shape: [H, W, C]).
    full_lr_np = full_lr.permute(1, 2, 0).numpy()
    full_sr_np = full_sr.permute(1, 2, 0).numpy()

    # Create a side-by-side plot: left = full LR, right = full super-resolved.
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(full_lr_np)
    axs[0].set_title("Full Low Resolution")
    axs[0].axis("off")

    axs[1].imshow(full_sr_np)
    axs[1].set_title("Full Super Resolved")
    axs[1].axis("off")

    axs[2].imshow(full_sr_np)
    axs[2].set_title("Original High Resolution")
    axs[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved full image figure to {output_path}")

# ------------------------------------------------------------------------------
# Example usage:
#
# patch_dir = "/path/to/your/high_res_patches"
# output_path = "stitched_full_image.png"
# image_id = "0052"   # if you want to process only patches for image "0052"
#
# # Ensure your model and sample() function are loaded/defined.
# process_hr_patches_and_save_full_image(patch_dir, model, device="cuda",
#                                        output_path=output_path, image_id=image_id,
#                                        hr_size=128, lr_size=32)