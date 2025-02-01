import torchshow as ts
import torchvision.utils as vutils

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time

def sample(model, lr_img, device="cuda"):
    model.to(device)
    model.eval()

    stime = time.time()
    with torch.no_grad():
        y = torch.randn_like(lr_img, device=device)
        lr_img = lr_img.to(device)

        for i, t in enumerate(range(model.time_steps - 1, 0, -1)):
            alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
            t = torch.tensor(t, device=device).long()
            pred_noise = model(torch.cat([lr_img, y], dim=1), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1/alpha_t)) * (y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise

    ftime = time.time()

    # Select only the first image from the batch (shape: [3, 128, 128])
    img = y[0].detach().cpu()  # Take the first image
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()  # Convert to HWC format

    # Show the first image using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.title("Denoised Image Output (First in Batch)")
    plt.show()
    # Select only the first image from the batch (shape: [3, 128, 128])
    img = lr_img[0].detach().cpu()  # Take the first image
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()  # Convert to HWC format

    print(f"Done denoising in {ftime - stime}s, image saved as 'sr_sample.jpeg'")
    # Function to display an image

def show_image(img_tensor, title):
    img = img_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()

def process_batch(loader, model, device="cuda"):

    """
    Process a single batch from the data loader.

    - Prints the shape of the LR and HR images.
    - Runs the model on the LR images.
    - Displays the first LR and HR images from the batch.

    Parameters:
    - loader: DataLoader providing batches of (LR images, HR images)
    - ddpm: Model used for processing the LR images
    - device: Device to run the computations on ("cuda" or "cpu")
    """
    ddpm = model
    # Get one batch of LR and HR images
    for lr_img, hr_img in loader:
        print(f"LR shape: {lr_img.shape}, HR shape: {hr_img.shape}")
        break  # Only process the first batch

    # Move LR images to the specified device
    lr_img = lr_img.to(device)

    # Run model sampling
    sample(model = ddpm, lr_img = lr_img,device = device)
    # Display the first LR image from the batch
    show_image(lr_img[0], "LR Image Output (First in Batch)")

    # Display the first HR image from the batch
    show_image(hr_img[0], "HR Image Output (First in Batch)")