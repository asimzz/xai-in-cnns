import torch
import numpy as np

class Cutout:
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        h, w = img.size(1), img.size(2)
        mask_size_half = self.mask_size // 2

        cx = np.random.randint(mask_size_half, w - mask_size_half)
        cy = np.random.randint(mask_size_half, h - mask_size_half)

        mask = np.ones((h, w), np.float32)

        y1 = np.clip(cy - mask_size_half, 0, h)
        y2 = np.clip(cy + mask_size_half, 0, h)
        x1 = np.clip(cx - mask_size_half, 0, w)
        x2 = np.clip(cx + mask_size_half, 0, w)

        mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        img *= mask

        return img
