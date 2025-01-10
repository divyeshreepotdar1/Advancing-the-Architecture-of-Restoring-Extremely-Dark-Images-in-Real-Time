import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import glob
from common_classes1 import load_data, run_test
from network import Net , Autoencoder
import matplotlib.pyplot as plt
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

save_images = 'demo_restored_images'

shutil.rmtree(save_images, ignore_errors = True)
os.makedirs(save_images)

test_files = glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Original/train_test_ours/demo_imgs/*.ARW') 
dataloader_test = DataLoader(load_data(test_files), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = Net()
print('\n Network parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on GPU: {}'.format(next(model.parameters()).is_cuda))
checkpoint = torch.load('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Original/train_test_ours/weights/weights_10000', map_location=device)
model.load_state_dict(checkpoint['model'])

run_test(model, dataloader_test, save_images)
print('Restored images saved in DEMO_RESTORED_IMAGES directory')

'''
loss_function = nn.CrossEntropyLoss()
model.eval()  # Set the model to evaluation mode
test_loss = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():  # No need to track gradients for evaluation
    for data, target in dataloader_test:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        # Calculate loss (assuming you have a loss function like nn.CrossEntropyLoss)
        loss = loss_function(output, target)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

test_loss /= len(dataloader_test.dataset)
accuracy = 100. * correct_predictions / total_predictions
print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')'''
