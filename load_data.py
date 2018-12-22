# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# A function that loads and preprocesses the flower image datasets for training, validation, and testing

import torch
from torchvision import datasets, transforms, models


def load_data(image_dir):
    
    train_dir = image_dir + '/train'
    valid_dir = image_dir + '/valid'
    test_dir = image_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_image_dataset = datasets.ImageFolder(test_dir ,transform=test_transforms)
    valid_image_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    class_to_idx = train_image_dataset.class_to_idx
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_image_dataset, batch_size=20, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=32,shuffle=True)

    return class_to_idx, train_loader, test_loader, valid_loader
