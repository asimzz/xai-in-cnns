import shutil
from torchvision import transforms
from os import makedirs, listdir
from os.path import join, isdir, isfile
from sklearn.model_selection import train_test_split


def split_dataset_folder(local_path: str = "/dataset"):
    data_dir = f"{local_path}/flowers"
    train_dir = f"{local_path}/train"
    test_dir = f"{local_path}/test"


    makedirs(train_dir, exist_ok=True)
    makedirs(test_dir, exist_ok=True)

    classes = [d for d in listdir(data_dir) if isdir(join(data_dir, d))]

    for cls in classes:
        makedirs(join(train_dir, cls), exist_ok=True)
        makedirs(join(test_dir, cls), exist_ok=True)


        class_dir = join(data_dir, cls)
        images = [f for f in listdir(class_dir) if isfile(join(class_dir, f))]

        train_images, test_images = train_test_split(
            images, test_size=0.2, random_state=42
        )

        # Move the images
        for img in train_images:
            shutil.move(join(class_dir, img), join(train_dir, cls, img))
        for img in test_images:
            shutil.move(join(class_dir, img), join(test_dir, cls, img))

    print("Dataset split into train and test folders successfully :).")

    return train_dir, test_dir


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(336),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return train_transforms, test_transforms
