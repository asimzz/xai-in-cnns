import matplotlib.pyplot as plt

def display_images(image, heatmap, cam, target, prediction):
    _, axes = plt.subplots(1, 3, figsize=(10, 5))
    image = image.permute(1, 2, 0).numpy()
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image, {target}")
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].axis('off')

    axes[2].imshow(cam)
    axes[2].set_title(f"CAM Image, {prediction}")
    axes[2].axis('off')
    plt.show()