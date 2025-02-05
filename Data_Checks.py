import random
from torchvision.transforms import transforms
import os, cv2
from PIL import Image



# -----------------------------------------
# inspection function
# -----------------------------------------
def inspect_dataset_output(data_loader):
    """
    Inspects the first batch of a given DataLoader by:
      - Retrieving a batch (lr_images, hr_images).
      - Applying an inverse transform to convert them back to PIL images.
      - Displaying the first image pair's size and plotting them.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader that returns (lr_images, hr_images).
    """
    # Fetch the first batch from the DataLoader
    batch = next(iter(data_loader))

    # If the batch is a list of tuples, extract the first element.
    if isinstance(batch, list) and len(batch) == 1:
        batch = batch[0]

    # Make sure we have a tuple (lr_images, hr_images)
    if not (isinstance(batch, tuple) and len(batch) == 2):
        raise ValueError(f"Expected (lr_images, hr_images), but got {type(batch)} with length {len(batch)}")

    lr_images, hr_images = batch

    # Take the first image from the batch and move to CPU if necessary
    lr_image = lr_images[0].cpu()
    hr_image = hr_images[0].cpu()

    # Define an inverse transform (if you want to undo normalization)
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5]),
        transforms.ToPILImage()
    ])

    lr_image_pil = inverse_transform(lr_image)
    hr_image_pil = inverse_transform(hr_image)

    print(f"Low-resolution image size: {lr_image_pil.size}")
    print(f"High-resolution image size: {hr_image_pil.size}")

    plt.figure(figsize=(8, 8))
    plt.title("High Resolution Image")
    plt.imshow(hr_image_pil)
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title("Low Resolution Image")
    plt.imshow(lr_image_pil)
    plt.axis('off')
    plt.show()
    # # Define the inverse normalization transformation (assuming mean=0.5, std=0.5 in original normalization)
    # inverse_transform = transforms.Compose([
    #     transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5]),  # Undo the normalization
    #     transforms.ToPILImage()  # Convert back to a PIL image
    # ])
    #
    # # Grab the first batch (lr_images, hr_images)
    # for lr_images, hr_images in data_loader:
    #     # Take the first image from the batch
    #     lr_image = lr_images[0]
    #     hr_image = hr_images[0]
    #
    #     # Un-normalize and convert to PIL image
    #     lr_image_pil = inverse_transform(lr_image)
    #     hr_image_pil = inverse_transform(hr_image)
    #
    #     # Display size info
    #     print(f"Low-resolution image size: {lr_image_pil.size}")
    #     print(f"High-resolution image size: {hr_image_pil.size}")
    #
    #     # Display images
    #     plt.figure(figsize=(8, 8))
    #     plt.title("High Resolution Image")
    #     plt.imshow(hr_image_pil)
    #     plt.axis('off')
    #     plt.show()
    #
    #     plt.figure(figsize=(8, 8))
    #     plt.title("Low Resolution Image")
    #     plt.imshow(lr_image_pil)
    #     plt.axis('off')
    #     plt.show()

        # Break so we only display the first batch

def inspect_data_example(image_dir, image_list=None, example_filename=None):
    """
    Display and print info for a single example image from your dataset.

    Args:
        image_dir (str): The directory where images are stored.
        image_list (list, optional): A list of image filenames.
                                     If provided and `example_filename` is None,
                                     the function chooses a random file from this list.
        example_filename (str, optional): If provided, tries to load this specific filename.
                                          Otherwise, picks one randomly (from `image_list` if given,
                                          or from `image_dir` if `image_list` is not given).
    """
    # 1) Pick an image: if example_filename is given, use that;
    #    else, sample randomly from image_list or the directory.
    if example_filename is not None:
        chosen_image = example_filename
    else:
        if image_list:
            chosen_image = random.choice(image_list)
        else:
            all_images = os.listdir(image_dir)
            chosen_image = random.choice(all_images)

    # 2) Form the full path
    example_image_path = os.path.join(image_dir, chosen_image)

    # 3) Load and display the image
    try:
        image = Image.open(example_image_path)
        plt.figure(figsize=(16, 16))
        plt.imshow(image)
        plt.axis('off')  # Hide axis labels
        plt.show()

        # 4) Print out the image size
        print(f"Displayed image: {chosen_image}")
        print(f"Size (W x H): {image.size}")
    except Exception as e:
        print(f"Error opening image '{example_image_path}': {e}")


import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def inspect_dataset_output(data_loader):
    """
    Inspects the first batch of a given DataLoader by:
      - Retrieving a batch (lr_images, hr_images).
      - Applying an inverse transform to convert them back to PIL images.
      - Displaying the first image pair's size and plotting them.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader that returns (lr_images, hr_images).
    """

    # Define the inverse normalization transformation (assuming mean=0.5, std=0.5 in original normalization)
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5]),  # Undo the normalization
        transforms.ToPILImage()  # Convert back to a PIL image
    ])

    # Fetch the first batch from the DataLoader
    batch = next(iter(data_loader))

    # If the batch is a list of length 2, accept it.
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        lr_images, hr_images = batch
    else:
        raise ValueError(f"Expected (lr_images, hr_images), but got {type(batch)} with length {len(batch)}")


    # Take the first image from the batch
    lr_image = lr_images[0]
    hr_image = hr_images[0]

    # Ensure tensors are on CPU before processing
    if lr_image.is_cuda:
        lr_image = lr_image.cpu()
    if hr_image.is_cuda:
        hr_image = hr_image.cpu()

    # Un-normalize and convert to PIL image
    lr_image_pil = inverse_transform(lr_image)
    hr_image_pil = inverse_transform(hr_image)

    # Display size info
    print(f"Low-resolution image size: {lr_image_pil.size}")
    print(f"High-resolution image size: {hr_image_pil.size}")

    # Display images
    plt.figure(figsize=(8, 8))
    plt.title("High Resolution Image")
    plt.imshow(hr_image_pil)
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title("Low Resolution Image")
    plt.imshow(lr_image_pil)
    plt.axis('off')
    plt.show()

