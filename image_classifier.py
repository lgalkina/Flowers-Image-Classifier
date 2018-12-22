# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# Classifies flower images using a pretrained CNN model

from get_input_args import get_input_args_to_train
from load_data import load_data
from build_network import build_network
from train import train
from process_checkpoint import save_checkpoint

def main():
    print("Image classifier starts")
    
    # retrieve command line arugments
    in_arg = get_input_args_to_train()
    
    # load and preprocess the flower image datasets for training, validation, and testing
    class_to_idx, train_loader, test_loader, valid_loader = load_data(in_arg.dir)    
    print("Data for training, validation, and testing is loaded")
    
    # define neural network
    model, criterion, optimizer = build_network(class_to_idx, in_arg.arch, in_arg.dropout, in_arg.learning_rate)    
    print("Network is built")
    
    # train neural network
    train(model, train_loader, valid_loader, criterion, optimizer, in_arg.epochs, in_arg.process_unit)    
    print("Network is trained")
    
    # save checkpoint
    save_checkpoint(in_arg.arch, in_arg.learning_rate, in_arg.epochs, in_arg.dropout, model, in_arg.checkpoint_path)    
    print("Checkpoint is saved")
    
if __name__ == "__main__":
    main()