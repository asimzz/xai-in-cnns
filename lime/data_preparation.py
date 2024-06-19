import os
from sklearn.model_selection import train_test_split
import shutil

def prepare_data(data_dir, train_dir, val_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        cls_dir = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

        if len(images) == 0:
            print(f"No images found for class {cls}. Skipping...")
            continue

        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

        for img in train_images:
            shutil.move(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.move(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))

    print("Dataset split into train and validation folders successfully.")
