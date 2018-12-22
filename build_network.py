# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# A function that prepares a neural network

import torchvision.models as models
from torch import optim
from torch import nn

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

nn_models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# Hyperparameters for network
input_sizes = {'resnet': 1024, 'alexnet': 9216, 'vgg': 25088}
output_size = 102


def build_network(class_to_idx, hidden_units, model_name = 'vgg', dropout = 0.5, learning_rate = 0.001):
    
    # check if provided model is supported
    if model_name not in nn_models.keys():
        raise ValueError('Unexpected network architecture', model_name)

    # model
    model = nn_models[model_name]

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # prepare layers
    linear_layers = [nn.Linear(input_sizes[model_name], hidden_units[0])]
    layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
    linear_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    layers = []
    for i in range(len(linear_layers)):
        layers.append(linear_layers[i])
        layers.append(nn.Dropout(dropout))
        # Rectified Linear Unit Function as activation function
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_units[-1], output_size))
    layers.append(nn.LogSoftmax(dim=1))
    layers = nn.ModuleList(layers)
    classifier = nn.Sequential(*layers)
    model.classifier = classifier

    criterion = nn.NLLLoss()
    # use Adam as optimizer
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.class_to_idx = class_to_idx
    
    return model, criterion, optimizer
