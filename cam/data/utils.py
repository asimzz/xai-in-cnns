import shutil
import zipfile
from os import makedirs, listdir, rmdir
from os.path import join, isdir, isfile, exists
from sklearn.model_selection import train_test_split


def split_dataset_folder(local_path: str = "/dataset"):
    data_dir = f"{local_path}/flowers"
    train_dir = f"{local_path}/train"
    val_dir = f"{local_path}/val"
    test_dir = f"{local_path}/test"

    # Create train, validation, and test directories
    makedirs(train_dir, exist_ok=True)
    makedirs(val_dir, exist_ok=True)
    makedirs(test_dir, exist_ok=True)

    # Get the list of classes (subfolders in the data_dir)
    classes = [d for d in listdir(data_dir) if isdir(join(data_dir, d))]

    for cls in classes:
        # Create class subdirectories in train, validation, and test directories
        makedirs(join(train_dir, cls), exist_ok=True)
        makedirs(join(val_dir, cls), exist_ok=True)
        makedirs(join(test_dir, cls), exist_ok=True)

        # Get the list of image files for the current class
        cls_dir = join(data_dir, cls)
        images = [f for f in listdir(cls_dir) if isfile(join(cls_dir, f))]

        # Split the images into train, validation, and test sets
        train_images, test_images = train_test_split(
            images, test_size=0.2, random_state=42
        )
        train_images, val_images = train_test_split(
            train_images, test_size=0.2, random_state=42
        )

        # Move images to the respective directories
        for img in train_images:
            shutil.move(join(cls_dir, img), join(train_dir, cls, img))
        for img in val_images:
            shutil.move(join(cls_dir, img), join(val_dir, cls, img))
        for img in test_images:
            shutil.move(join(cls_dir, img), join(test_dir, cls, img))

    print("Dataset split into train, validation, and test folders successfully.")

    return train_dir, val_dir, test_dir


def unzip_dataset(local_path: str):
    # # Remove the existing /celeba directory if it exists
    # rmdir(local_path)

    # Create the directory
    makedirs(local_path, exist_ok=True)

    # Path to the zip file in your Google Drive
    zip_file_path = join(local_path, "flowers.zip")

    # Check if the zip file exists
    if exists(zip_file_path):
        # Unzip the file into the /flowers directory
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(local_path)
            print("Unzipping completed.")
    else:
        print("Zip file does not exist.")
