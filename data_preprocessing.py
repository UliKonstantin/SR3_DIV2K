import json

from DataLoader import *
from DataLoader import inspect_data_example, inspect_dataset_output


# If you need image transformations:


def preprocess_data(config_path="config.json"):
    """
    Reads the config.json file, splits the dataset, inspects a few examples,
    creates the dataset and dataloader, and returns them.
    """
    # 1) Load config values
    with open(config_path, "r") as f:
        config = json.load(f)

    train_hr_dir = config["train_hr_dir"]
    valid_hr_dir = config["valid_hr_dir"]
    test_sample_size = config["test_sample_size"]
    crop_size = tuple(config["crop_size"])  # e.g. (128, 128)

    # 2) Split data
    train_images, test_images, val_images = split_data(
        train_hr_dir=train_hr_dir,
        valid_hr_dir=valid_hr_dir,
        test_sample_size=test_sample_size
    )

    # 3) Inspect some data (optional but helpful)
    inspect_data_example(train_hr_dir, image_list=None, example_filename=None)

    # 4) Create dataset
    train_sr_dataset = SRDataset(
        dataset_path=train_hr_dir,
        crop_size=crop_size,
        # If you have custom transforms, add them here
        #_transforms = transforms.Compose([
        #        transforms.ToTensor()])
    )

    # 5) Create DataLoader
    train_sr_loader = DataLoader(
        train_sr_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # Inspect dataset output (optional)
    inspect_dataset_output(train_sr_loader)

    # Return the dataset and loader for training
    return train_sr_dataset, train_sr_loader