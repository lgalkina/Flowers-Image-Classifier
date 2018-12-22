# Flowers-Image-Classifier
This project is part of [Udacity](https://www.udacity.com "Udacity - Be in demand")'s [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).

The goals of this project is to classify flower images using a pretrained CNN model.

## Versions
- [Anaconda 5.3 with Python 3](https://www.anaconda.com/download/)
- [PyTorch 0.4.1](https://pytorch.org/)

## Command Line Application

To train a new network on a data set:

- Run: python image_classifier.py 
- The current epoch, training loss, validation loss, and validation accuracy as the netowrk trains will be printed
- Command line arguments: 
    - Image folder as --dir with default value './flowers'
    - CNN model architecture as --arch with default value 'vgg'
    - Learning rate as --learning_rate with default value 0.001
    - Dropout as --dropout with default value 0.5
    - Number of epochs as --epochs with default value 20
    - Path to save checkpoint to as --checkpoint_path with default value './checkpoint.pth'
    - Processing unit --process_unit with default value 'gpu'
    
 To predict flower name from an image:
 - Run: python image_classifier.py 
 - Command line arguments: 
    - Path to image to predict as --image with default value './flowers/test/1/image_06764.jpg'
    - Path to retrieve checkpoint to as --checkpoint_path with default value './checkpoint.pth'
    - Top K predictions as --topk with default value 5
    - Path to JSON file containing label names as --labels with default value './cat_to_name.json'
    - Processing unit --process_unit with default value 'gpu'
