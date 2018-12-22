# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# A function that returns top k number of most probable choices that the network predicts

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def predict(image_path, model, topk = 5, process_unit = 'gpu'):
    
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

