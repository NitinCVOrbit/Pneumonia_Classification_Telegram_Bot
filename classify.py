from torchvision.transforms import transforms
import torch
import numpy as np


def preprocessing(image):
    """Preprocesses an image for classification."""
    transformation = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])
    img = transformation(image)
    img = img.unsqueeze(0)
    return img


def classifier(image, model, class_names):    
    """Classifies an image as Brain Tumor or No Brain Tumor."""
    preprocessed_img = preprocessing(image)
    model.eval()    
    output = model(preprocessed_img)
    pred = torch.argmax(output, dim=1).item()  # Get predicted class index
    label = class_names[pred]                 # Map index to class label
    score = int(torch.round(output[0][pred] * 100).item())  # Confidence in %
    
    return label,score