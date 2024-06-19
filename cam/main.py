import torch
from os.path import join
from torchvision import transforms

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import ResNet18WithCAM, CAMVisualizer
from data import FlowersDataset, split_dataset_folder, unzip_dataset
from utils import train, predict, get_device
from occulusions import Cutout
from plots import display_images



origin_path = "/Users/asim-abdalla/AMMI/projects/xai-in-cnns"
local_path = join(origin_path, "dataset")
batch_size = 32

unzip_dataset(local_path)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(336),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(mask_size=100, p=0.75)
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(mask_size=100, p=0.75)
        ]
    )

    return train_transforms, test_transforms


train_dir, val_dir, test_dir = split_dataset_folder(local_path)
train_transforms, test_transforms = get_transforms()

train_data = FlowersDataset(path=train_dir, transform=train_transforms)
val_data = FlowersDataset(path=val_dir, transform=train_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                 drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                                 drop_last=True)

print(train_data)

device, cuda = get_device()
classifier = ResNet18WithCAM(num_classes=train_data.num_classes)


num_epochs = 1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)

train(classifier, optimizer, criterion,scheduler, train_loader, val_loader, device, epochs=num_epochs)


# checkpoint = torch.load(f"{origin_path}/resnet18_model.pth")
# classifier.to(device)
# classifier.load_state_dict(checkpoint)


test_dataset = FlowersDataset(path=test_dir, transform=test_transforms)
len(test_dataset)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)

predict(classifier, test_loader, device)


visualizer = CAMVisualizer(classifier)

for i, (images, target) in enumerate(test_loader):
    if i % 2 == 0:  # Apply CAM to first 10 images in the test dataset
        continue
    idx = i % 32
    images = images.to(device)
    output = classifier(images)
    pred = torch.argmax(output, dim=1).cpu().numpy()[idx]
    target = target.cpu().numpy()[idx]

    feature_maps = classifier.feature_maps.cpu().detach().numpy()[idx]

    image = images[idx].cpu()
    visualizer.generate_cam(feature_maps, pred)
    visualizer.generate_image_heatmap(image)
    labels = test_dataset.classes_labels
    display_images(image, visualizer.heatmap, visualizer.cam, labels[target], labels[pred])
