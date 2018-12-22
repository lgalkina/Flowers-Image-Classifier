# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# Function that save and load checkpoint

import torch
from build_network import build_network


# save checkpoint
def save_checkpoint(arch, learning_rate, epochs, dropout, hidden_units, model, checkpoint_path = './checkpoint.pth'):
    checkpoint = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    torch.save(checkpoint, checkpoint_path)


# load checkpoint
def load_checkpoint(checkpoint_path = './checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    # define neural network
    model, optimizer, criterion = build_network(checkpoint['class_to_idx'], checkpoint['hidden_units'], checkpoint['arch'], checkpoint['dropout'], checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
