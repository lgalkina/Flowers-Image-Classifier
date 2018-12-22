# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# A function that prepares a neural network

import torchvision.models as models
from torch import optim
from torch import nn
from collections import OrderedDict

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

nn_models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# Hyperparameters for network
input_sizes = {'resnet': 1024, 'alexnet': 9216, 'vgg': 25088} 
hidden_sizes = [512, 102]

def build_network(class_to_idx, model_name = 'vgg', dropout = 0.5, learning_rate = 0.001):
    
    # check if provided model is supported
    if model_name not in nn_models.keys():
        raise ValueError('Unexpected network architecture', model_name)

    # model
    model = nn_models[model_name]

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # apply the Rectified Linear Unit Function as activation function
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_sizes[model_name], hidden_sizes[0])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    # use Adam as optimizer
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.class_to_idx = class_to_idx
    
    return model, criterion, optimizer