import os
import cv2
import json
import torch
from DataLoader import *  # or your relevant imports

limit = -1
crop_size = (128, 128)
stride = None

with open("/Users/UliKonstantin4/PycharmProjects/SR3/config.json", "r") as f:
    config = json.load(f)

train_hr_dir = '/Users/UliKonstantin4/PycharmProjects/SR3/DIV2K/DIV2K_train_HR/'
valid_hr_dir = '/Users/UliKonstantin4/PycharmProjects/SR3/DIV2K/DIV2K_valid_HR/'
test_sample_size = config["test_sample_size"]
crop_size = tuple(config["crop_size"])  # e.g. (128, 128)

# 2) Split data
train_images, test_images, val_images = split_data(
    train_hr_dir=train_hr_dir,
    valid_hr_dir=valid_hr_dir,
    test_sample_size=test_sample_size
)

augment_directory = '/Users/UliKonstantin4/PycharmProjects/SR3/DIV2K/Augment_Train'
# Make sure the directory exists
os.makedirs(augment_directory, exist_ok=True)

# If stride is None, default = crop_size => no overlap
stride = stride if stride is not None else crop_size[0]

# Collect valid image paths
valid_extensions = {"jpg", "jpeg", "png", "JPEG", "JPG"}
all_files = os.listdir(train_hr_dir)
if limit > 0:
    all_files = all_files[:limit]
image_paths = [
    os.path.join(train_hr_dir, f)
    for f in all_files
    if f.split(".")[-1] in valid_extensions
]

crop_h, crop_w = crop_size

# Loop over each image and save patches
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    # If the entire image is smaller than (crop_h x crop_w), skip
    if h < crop_h or w < crop_w:
        continue

    # Extract the basename (e.g., "0001" from "/path/to/0001.png")
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # For each top-left corner => (x, y)
    for y in range(0, h - crop_h + 1, stride):
        for x in range(0, w - crop_w + 1, stride):
            # Crop the patch as a NumPy array
            patch_img = img[y: y + crop_h, x: x + crop_w]

            # Build a filename like: "0001_0_0.png" for the patch
            patch_name = f"{base_name}_{x}_{y}.png"
            patch_path = os.path.join(augment_directory, patch_name)

            # Save the patch
            cv2.imwrite(patch_path, patch_img)

print("All patches have been saved to:", augment_directory)
