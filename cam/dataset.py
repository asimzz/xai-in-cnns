import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class FlowersDataset(Dataset):
    def __init__(self, path="/dataset", transform=None):
      self.transform = transform

      dataset_path = Path(path).resolve()
      self.paths = list(dataset_path.glob("*/*"))
      self.classes = [path.parent.stem for path in self.paths]
      self.num_classes = len(set(self.classes))

      dataset_frame = pd.DataFrame({"path": self.paths, "class": self.classes})
      dataset_frame["class"] = dataset_frame["class"].astype("category")
      dataset_frame["label"] = dataset_frame["class"].cat.codes

      self.labels = dict(zip(dataset_frame["class"].cat.categories, range(self.num_classes)))
      self.classes_labels = dict(zip(range(self.num_classes),dataset_frame["class"].cat.categories))


    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        class_label = self.labels[self.classes[idx]]
        im = Image.open(image_path)
        im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        return im, class_label