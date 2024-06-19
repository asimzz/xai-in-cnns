import numpy as np
import torch

class RandomErase:
    def __init__(self, min_mask_size, max_mask_size, p=0.5):
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        h, w = img.size(1), img.size(2)
        mask_height = np.random.randint(self.min_mask_size, self.max_mask_size)
        mask_width = np.random.randint(self.min_mask_size, self.max_mask_size)

        mask_height_half = mask_height // 2
        mask_width_half = mask_width // 2

        cx = np.random.randint(mask_width_half, w - mask_width_half)
        cy = np.random.randint(mask_height_half, h - mask_height_half)

        mask = torch.rand((h, w), dtype=torch.float32)

        y1 = np.clip(cy - mask_height_half, 0, h)
        y2 = np.clip(cy + mask_height_half, 0, h)
        x1 = np.clip(cx - mask_width_half, 0, w)
        x2 = np.clip(cx + mask_width_half, 0, w)

        img[:, y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        return img