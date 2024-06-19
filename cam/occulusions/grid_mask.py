import numpy as np
import torch

class GridMask:
    def __init__(self, d1, d2, ratio=0.5, p=0.5):
        self.d1 = d1
        self.d2 = d2
        self.ratio = ratio
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        d = np.random.randint(self.d1, self.d2)
        l = int(d * self.ratio)

        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:i+l, j:j+l] = 0

        mask = torch.from_numpy(mask).expand_as(img)
        img *= mask

        return img
