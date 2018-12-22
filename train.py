# PROGRAMMER: Lyudmila Galkina
# DATE CREATED: 21.12.2018 
# Train and validate neural network

import torch
from PIL import Image

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, process_unit = 'gpu'):
    
    print_every = 40
    steps = 0
    
    # Use gpu if selected and available
    if process_unit == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.cuda()
    else:
        device = torch.device("cpu") 
    
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        # iterate over data
        for inputs, labels in iter(train_loader):                
            inputs, labels = inputs.to(device), labels.to(device)
            
            steps += 1
                
            # clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # forward processing
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            # backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                
                if process_unit == 'gpu' and torch.cuda.is_available():
                    model.cuda()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, valid_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
        
# validate
def validate(model, valid_loader, criterion, device):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in iter(valid_loader):      
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy