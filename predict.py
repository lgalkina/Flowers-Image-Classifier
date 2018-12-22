# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# Provides image predictions

import json
import torch
import torch.nn.functional as F
from get_input_args import get_input_args_to_predict
from process_checkpoint import load_checkpoint
from torchvision import transforms
from PIL import Image


# a function that returns top k number of most probable choices that the network predicts
def predict(image_path, model, topk=5, process_unit='gpu'):
    
    if process_unit == 'gpu' and torch.cuda.is_available():
        model.cuda()
    
    model.eval()
    
    image = process_image(image_path)
    # this is for VGG
    image = image.unsqueeze(0) 
    image = image.float()
    
    if process_unit == 'gpu' and torch.cuda.is_available():
        image = image.cuda()
            
    with torch.no_grad():
        output = model.forward(image)

    probability = F.softmax(output.data, 1)

    return probability.topk(topk)


# open and trasform image and return it as a tensor
def process_image(image_path):
    image = Image.open(image_path)

    # transform image
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    tensor_image = transform_image(image)
    
    return tensor_image


def main():
    print("Image predictor starts")
    
    # retrieve command line arugments
    in_arg = get_input_args_to_predict()
    
    # load checkpoint
    model = load_checkpoint(in_arg.checkpoint_path)    
    print("Checkpoint is loaded")
    
    # predictions for image
    probabilities, classes = predict(in_arg.image, model, in_arg.topk, in_arg.process_unit)
    
    # load labels
    with open(in_arg.labels, 'r') as json_file:
        cat_to_name = json.load(json_file)
    
    class_names = [cat_to_name[str(x.item() + 1)] for x in classes[0]]
    print('Flower: {}'.format(class_names[0]))
    print('Probability: {}'.format(probabilities[0][0]))
    
    print(class_names)
    print(probabilities[0])


if __name__ == "__main__":
    main()
