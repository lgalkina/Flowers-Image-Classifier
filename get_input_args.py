# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# A function that retrieves command line inputs 
# from the user using the Argparse Python module to be used for neural network training. 
# If the user fails to provide some or all of the 8 inputs, then the default values are
# used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value './flowers'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Learning Rate as --learning_rate with default value 0.001
#     4. Dropout as --dropout with default value 0.5
#     5. Number of Epochs as --epochs with default value 20
#     6. Path to save checkpoint to as --checkpoint_path with default value './checkpoint.pth'
#     7. Processing unit --process_unit with default value 'gpu'

import argparse

def get_input_args_to_train():
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
        
    # Create 7 command line arguments using add_argument() from ArguementParser method
    parser.add_argument('--dir', type=str, default='./flowers/', help='path to image folder')
    parser.add_argument('--arch', type=str, default='vgg', help='name of CNN model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint.pth", help='checkpoint path')
    parser.add_argument('--process_unit', type=str, default="gpu", help='processing unit')
    
    return parser.parse_args()

# A function that retrieves command line inputs 
# from the user using the Argparse Python module to be used for image prediction. 
# If the user fails to provide some or all of the 8 inputs, then the default values are
# used for the missing inputs. Command Line Arguments:
#     1. Path to image to predict as --image with default value './flowers/test/1/image_06764.jpg'
#     2. Path to retrieve checkpoint to as --checkpoint_path with default value './checkpoint.pth'
#     3. Top K predictions as --topk with default value 5
#     4. Path to JSON file containing label names as --labels with default value './cat_to_name.json'
#     5. Processing unit --process_unit with default value 'gpu'
def get_input_args_to_predict():
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
        
    # Create 5 command line arguments using add_argument() from ArguementParser method
    parser.add_argument('--image', type=str, default='./flowers/test/1/image_06752.jpg', help='path to image to predict')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint.pth', help='checkpoint path')
    parser.add_argument('--topk', type=int, default=5, help='top K predictions')
    parser.add_argument('--labels', type=str, default='./cat_to_name.json', help='JSON file containing label names')
    parser.add_argument('--process_unit', type=str, default="gpu", help='processing unit')
    
    return parser.parse_args()