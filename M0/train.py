'''
Since our initial experiments last year and for patenting we have slightly improved the network. Like instead of ranging 'm' 
from 0.1 to 0.9 it now ranges from 1 to 9. We also found that modern CUDA libraries have been specially designed for 3x3 kernels 
and by design larger kernels will be slower. Thus we additionally discourage use of large kernels. Further 
'Learning Raw Image Denoising with Bayer Pattern Unification and Bayer Preserving Augmentation' suggests special data 
augmentations for Bayer pattern. Use them for better restoration. For better performance use different learning rate for 
conv. weights and bias (https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch/blob/470a6a8e9c6367d8fa88ee6d1dea211dd9fb1f81/demo.py#L146).
Finally depending on amount of CPU RAM you have, choose the number of images to be loaded in CPU RAM.
'''

opt={'base_lr':1e-4} # Initial learning rate
opt['reduce_lr_by'] = 0.1 # Reduce learning rate by 10 times
opt['atWhichReduce'] = [50000] # Reduce learning rate at these iterations.
opt['batch_size'] = 32
opt['atWhichSave'] = [2, 2000, 4000, 6000, 8000, 10000, 20000, 100000, 150002,200000,250002,300002,350002,400002,450002,500002,550000, 600000,650002,700002,750000,800000,850002,900002,950000,1000000] # testing will be done at these iterations and corresponding model weights will be saved.
opt['iterations'] = 10001 # The model will run for these many iterations.
dry_run = False # If you wish to first test the entire workflow, for couple of iterations, make this TRUE
dry_run_iterations = 1000 # If dry run flag is set TRUE the code will terminate after these many iterations

metric_average_file = 'metric_average.txt' # Average metrics will be saved here. Please note these are only for supervison. We used MATLAB for final PSNR and SSIM evaluation.
test_amplification_file = 'test_amplification.txt' # Intermediate details for the test images, such as estimated amplification will be saved here.
train_amplification_file = 'train_amplification.txt' # Intermediate details for the train images, such as estimated amplification will be saved here.

# These are folders
save_weights = 'weights' # Model weights will be saved here.
save_images = 'images' # Restored images will be saved here.
save_csv_files = 'csv_files' # Other details such as loss value and learning rate will be saved in this file.

import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from common_classes import load_data, run_test
from network import Net, Autoencoder
from vainF_ssim import MS_SSIM

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0" # You would probably like to change it to 0 or some other integer depending on GPU avalability.

shutil.rmtree(metric_average_file, ignore_errors = True)
shutil.rmtree(test_amplification_file, ignore_errors = True)
shutil.rmtree(train_amplification_file, ignore_errors = True)

shutil.rmtree(save_weights, ignore_errors = True)
shutil.rmtree(save_images, ignore_errors = True)
shutil.rmtree(save_csv_files, ignore_errors = True)

os.makedirs(save_weights)
os.makedirs(save_images)
os.makedirs(save_csv_files)

# train_files = glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Restoring-Extremely-Dark-Images-In-Real-Time/Sony/short/0*_00_0.1s.ARW')
train_files = glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Restoring-Extremely-Dark-Images-In-Real-Time/Sony/Sony/short/0*_00_0.1s.ARW')
train_files +=glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Restoring-Extremely-Dark-Images-In-Real-Time/Sony/Sony/short/2*_00_0.1s.ARW')


if dry_run:
    train_files = train_files[:4]
    opt['iterations'] = dry_run_iterations
    
gt_files = []
for x in train_files:
    gt_files += glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Restoring-Extremely-Dark-Images-In-Real-Time/Sony/Sony/long/*'+x[-17:-12]+'*.ARW')
    
dataloader_train = DataLoader(load_data(train_files,gt_files,train_amplification_file,20,gt_amp=True,training=True), batch_size=opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

test_files = glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Restoring-Extremely-Dark-Images-In-Real-Time/Sony/Sony/short/1*_00_0.1s.ARW') 
if dry_run:
    test_files = test_files[:4]
    
gt_files = []
for x in test_files:
    gt_files = gt_files+ glob.glob('/media/robotixx/SSD_Data2/Aniket/DL747_Project/code2/Restoring-Extremely-Dark-Images-In-Real-Time/Sony/Sony/long/*'+x[-17:-12]+'*.ARW')
dataloader_test = DataLoader(load_data(test_files,gt_files,test_amplification_file,2,gt_amp=True,training=False), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


for i,img in enumerate(dataloader_train):    
    print('Input image size : {}, GT image size : {}'.format(img[0].size(), img[1].size()))    
    break
    
############ Training Begins

device = torch.device("cuda")
model = Net()
# model = Autoencoder()
print(model)
print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

iter_num = 0
l1_loss = torch.nn.L1Loss()
#calculate_accuracy = torch.nn.accuracy()
def psnr(pred, gt):
    # Calculate Mean Squared Error (MSE) between prediction and ground truth
    mse = F.mse_loss(pred, gt)
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    # Calculate PSNR
    PIXEL_MAX = 1.0  # assuming the images are scaled between 0 and 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#feature_loss = Vgg16().to(device)
dssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
optimizer.zero_grad()
loss_list = ['L1_loss,Feature_loss,MS_SSIM']
loss_iter_list = ['Iteration']
iter_LR = ['Iter_LR']
            
loss_values = []
accuracy_values = []
while iter_num<opt['iterations']:
    for _, img in enumerate(dataloader_train):
        low = img[0].to(device)
        gt = img[1].to(device)
        model.train()
        pred = model(low)
        iter_num +=1
        loss3 = 1-dssim(pred, gt)
        #loss2 = feature_loss(pred, gt, which='relu2')
        loss1 = l1_loss(pred,gt)
        #print('l1,vgg loss: ',loss1.item(), loss2.item())
        loss = loss1 + (0.2*loss3)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
        loss_values.append(loss.item())
        #print(loss_values)
        # Calculate accuracy (this is a placeholder, adapt it to your needs)
        accuracy = psnr(pred, gt)  # You need to define this function
        accuracy_values.append(accuracy)
        if iter_num>opt['iterations']:
            break        
        
        if iter_num%10==0:
            print(iter_num)
            if iter_num%100==0:
                loss_list.append('{},{},{}'.format(loss1.item(),-1,loss3.item()))
                loss_iter_list.append(iter_num)
                iter_LR.append(optimizer.param_groups[0]['lr'])
                
        if iter_num in opt['atWhichSave']:
            print('testing......')
            if iter_num == opt['atWhichSave'][0]:
                mode = 'w'
            else:
                mode = 'a'
            run_test(model, dataloader_test, iter_num, save_images, save_csv_files, metric_average_file, mode, training=True)
            torch.save({'model': model.state_dict()},os.path.join(save_weights,'weights_{}'.format(iter_num)))
            print(os.path.join(save_weights,'weights_{}'.format(iter_num)))
            
        if iter_num in opt['atWhichReduce']:
            for group in optimizer.param_groups:
                old_lr = group['lr']
                group['lr'] = opt['reduce_lr_by']*old_lr
                if group['lr']<1e-5:
                    group['lr']=1e-5
                print('Changing LR from {} to {}'.format(old_lr,group['lr']))
print("loss values",loss_values)
print("accuracy values",accuracy_values)  
np.savetxt(os.path.join(save_csv_files,'loss_curve.csv'),[p for p in zip(loss_iter_list,loss_list,iter_LR)],delimiter=',',fmt='%s')   

# Optionally, you can save the PSNR values to a file
with open('psnr_values.txt', 'w') as file:
    for value in accuracy_values:
        file.write(f'{value}\n')
# Optionally, you can save the PSNR values to a file
with open('loss_values.txt', 'w') as file:
    for value in loss_values:
        file.write(f'{value}\n')              
# Plotting Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_values, label='Loss')
plt.title('Training Loss')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy_values, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('iterations')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()               