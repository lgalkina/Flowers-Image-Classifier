# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# Provides image predictions

import json
from get_input_args import get_input_args_to_predict
from process_checkpoint import load_checkpoint
from predict import predict

def main():
    print("Image predictor starts")
    
    # retrieve command line arugments
    in_arg = get_input_args_to_predict()
    
    # load checkpoint
    model = load_checkpoint(in_arg.checkpoint_path)    
    print("Checkpoint is loaded")
    
    # load labels
    with open(in_arg.labels, 'r') as json_file:
        cat_to_name = json.load(json_file)
    
    # predictions for image
    probabilities, classes = predict(in_arg.image, model, in_arg.topk, in_arg.process_unit)     
    print("Probabilities: ", probabilities)
    print("Classes: ", classes)
    
if __name__ == "__main__":
    main()