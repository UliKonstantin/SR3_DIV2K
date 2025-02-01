import os
import random
from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms
import os, cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Set up paths for training and validation directories
train_hr_dir = 'DIV2K/DIV2K_train_HR/'
valid_hr_dir = 'DIV2K/DIV2K_valid_HR/'


def split_data(train_hr_dir, valid_hr_dir, test_sample_size=100):
    """
    Splits the train directory images into train and test subsets.

    Args:
        train_hr_dir (str): Path to the directory containing training HR images.
        valid_hr_dir (str): Path to the directory containing validation HR images.
        test_sample_size (int): How many images to sample for the test set.

    Returns:
        tuple: (train_images, test_images, val_images)
               Each is a list of filenames in their respective split.
    """
    # List the contents of each directory
    train_images = os.listdir(train_hr_dir)
    val_images = os.listdir(valid_hr_dir)

    # Show how many training images we have initially
    print(f"Train Before Splitting: {len(train_images)}")

    # Randomly sample a portion from the training set for test
    test_images = random.sample(train_images, test_sample_size)

    # Remaining train images are those not in the test set
    train_images = [img for img in train_images if img not in test_images]

    # Log final counts
    print(f"Train After Split: {len(train_images)}, Test: {len(test_images)}, Validation: {len(val_images)}")

    return train_images, test_images, val_images

class SRDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None,crop_size=(128, 128), hr_sz = 128, lr_sz = 32) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.transforms = _transforms

        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ColorJitter([0.5, 1]),
                transforms.RandomAdjustSharpness(1.1, p = 0.4),
                transforms.Normalize((0.5, ), (0.5,)) # normalizing image with mean, std = 0.5, 0.5
            ])
        self.hr_sz, self.lr_sz = transforms.Resize((hr_sz, hr_sz), interpolation=InterpolationMode.BICUBIC), transforms.Resize((lr_sz, lr_sz), interpolation=InterpolationMode.BICUBIC)

        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]

        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]
        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        # 2) Convert NumPy array to PIL so we can use torchvision transforms
        image_pil = Image.fromarray(image)

        # 3) Rotate if height > width (to match DIV2K approach)
        if h > w:
            image_pil = image_pil.rotate(-90, expand=True)

        # 4) Random crop (e.g., to 128×128) using torchvision
        image_pil = transforms.RandomCrop(self.crop_size)(image_pil)

        # 5) Apply the rest of the transforms (flip, color jitter, etc.)
        image_pil = self.transforms(image_pil)

        # 6) Split into HR and LR:
        #    - 'hr_image': resized to hr_sz×hr_sz
        #    - 'lr_image': then resized to lr_sz×lr_sz
        hr_image = self.hr_sz(image_pil)   # shape [3, hr_sz, hr_sz]
        lr_image = self.lr_sz(image_pil)   # shape [3, lr_sz, lr_sz]

        # 7) Downscale the LR image (just done above), then upsample back to hr_sz if needed
        #    But from your code, it looks like you want the final return to be:
        #    - x = upscaled LR (via 'self.hr_sz(lr_image)')
        #    - y = HR (hr_image)
        #    So the shape is still [3, hr_sz, hr_sz] for both, but x is lower-res data stretched up.

        return self.hr_sz(lr_image), hr_image
        # Here self.hr_sz(lr_image) => the "x" that is effectively a down->up scaled version,
        # while hr_image is the "y" ground truth.

class SRPatchDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 limit=-1,
                 _transforms=None,
                 crop_size=(128, 128),
                 hr_sz=128,
                 lr_sz=32,
                 stride=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.crop_size = crop_size
        self.hr_sz = hr_sz
        self.lr_sz = lr_sz

        # If stride is None, default = crop_size => no overlap
        self.stride = stride if stride is not None else crop_size[0]

        # Set up transforms
        if _transforms is not None:
            self.transforms = _transforms
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter([0.5, 1]),
                transforms.RandomAdjustSharpness(1.1, p=0.4),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.hr_resize = transforms.Resize((hr_sz, hr_sz), interpolation=transforms.InterpolationMode.BICUBIC)
        self.lr_resize = transforms.Resize((lr_sz, lr_sz), interpolation=transforms.InterpolationMode.BICUBIC)

        # Collect valid image paths
        valid_extensions = {"jpg", "jpeg", "png", "JPEG", "JPG"}
        all_files = os.listdir(dataset_path)
        if limit > 0:
            all_files = all_files[:limit]
        self.image_paths = [
            os.path.join(dataset_path, f)
            for f in all_files
            if f.split(".")[-1] in valid_extensions
        ]
        # Build a list of all patches
        self.patches = []
        self._build_patches()

    def _build_patches(self):
        crop_h, crop_w = self.crop_size
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape

            # If the entire image is smaller than (crop_h x crop_w), skip
            if h <= crop_h or w <= crop_w:
                continue

            # For each top-left corner => (x, y)
            for y in range(0, h - crop_h + 1, self.stride):
                for x in range(0, w - crop_w + 1, self.stride):
                    self.patches.append((img_path, x, y))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        img_path, x, y = self.patches[index]

        # Load via OpenCV => convert to PIL
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        # Crop the patch
        crop_h, crop_w = self.crop_size
        patch = img_pil.crop((x, y, x + crop_w, y + crop_h))

        # 5) Apply the rest of the transforms (flip, color jitter, etc.)
        patch = self.transforms(patch)

        # 6) Split into HR and LR:
        #    - 'hr_image': resized to hr_sz×hr_sz
        #    - 'lr_image': then resized to lr_sz×lr_sz
        # Resize for HR and LR
        hr_image = self.hr_resize(patch)
        lr_image = self.lr_resize(patch)

        # Upsample LR back to HR size
        upsampled_lr = self.hr_resize(lr_image)

        # 7) Downscale the LR image (just done above), then upsample back to hr_sz if needed
        #    But from your code, it looks like you want the final return to be:
        #    - x = upscaled LR (via 'self.hr_sz(lr_image)')
        #    - y = HR (hr_image)
        #    So the shape is still [3, hr_sz, hr_sz] for both, but x is lower-res data stretched up.

        return upsampled_lr, hr_image
        # Here self.hr_sz(lr_image) => the "x" that is effectively a down->up scaled version,
        # while hr_image is the "y" ground truth.
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

    # Define the inverse normalization transformation (assuming mean=0.5, std=0.5 in original normalization)
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5]),  # Undo the normalization
        transforms.ToPILImage()  # Convert back to a PIL image
    ])

    # Grab the first batch (lr_images, hr_images)
    for lr_images, hr_images in data_loader:
        # Take the first image from the batch
        lr_image = lr_images[0]
        hr_image = hr_images[0]

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

    # Grab the first batch (lr_images, hr_images)
    for lr_images, hr_images in data_loader:
        # Take the first image from the batch
        lr_image = lr_images[0]
        hr_image = hr_images[0]

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

        # Break so we only display the first batch
        break
