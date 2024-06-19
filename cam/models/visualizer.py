import cv2
import numpy as np


class CAMVisualizer:
    def __init__(
        self,
        classifier,
    ) -> None:
        params = list(classifier.resnet18.fc.parameters())
        self.weight_softmax = np.squeeze(params[0].cpu().detach().numpy())
        self.cam = None
        self.heatmap = None
        self.image_cam = None

    def generate_cam(self, feature_maps, class_idx):
        num_channels, height, width = feature_maps.shape
        feature_maps = feature_maps.reshape((num_channels, height * width))
        self.cam = self.weight_softmax[class_idx].dot(feature_maps)
        self.cam = self.cam.reshape(height, width)
        self.cam = self.cam - np.min(self.cam)
        self.cam = self.cam / np.max(self.cam)
        self.cam = np.uint8(255 * self.cam)

    def generate_image_heatmap(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min())
        self.image_cam = cv2.resize(self.cam, (image.shape[1], image.shape[0]))
        self.heatmap = cv2.applyColorMap(self.image_cam, cv2.COLORMAP_JET)
        self.heatmap = cv2.cvtColor(self.heatmap, cv2.COLOR_BGR2RGB)
        self.heatmap = np.float32(self.heatmap) / 255
        self.image_cam = self.heatmap + np.float32(image)
        self.image_cam = self.image_cam / np.max(self.image_cam)
