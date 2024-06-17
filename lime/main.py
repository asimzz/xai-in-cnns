import torch
from data_preparation import prepare_data
from data_transformation import get_dataloaders
from training import train_model
from inference import load_model, predict_image
from lime_explanation import explain_image

def main():
    # Define paths
    data_dir = '/Users/millicentomondi/Documents/xai-in-cnns/lime/flowers'
    base_dir = '/Users/millicentomondi/Documents/xai-in-cnns/lime'
    train_dir = f'{base_dir}/train'
    val_dir = f'{base_dir}/val'
    
    # Prepare data
    prepare_data(data_dir, train_dir, val_dir)
    
    # Get data loaders
    dataloaders, dataset_sizes, class_names = get_dataloaders(base_dir)
    
    # Determine device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    model = train_model(dataloaders, dataset_sizes, device)
    
    # Load the model for inference
    model = load_model(device)
    
    # Path to an unseen image
    image_path = '/Users/millicentomondi/Documents/XAI_GROUP/flower_images/tulip/1c8f7ee1bb.jpg'
    
    # Predict the class of the unseen image
    predict_image(model, image_path, class_names, device)
    
    # Generate LIME explanations for the image
    explain_image(model, image_path, class_names, device)

if __name__ == "__main__":
    main()
