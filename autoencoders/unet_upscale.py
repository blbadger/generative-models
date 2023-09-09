import math
from inspect import isfunction
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
import time
import pathlib
import numpy as np
import os
import random

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
import torchvision
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.optim import Adam

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from unet_noresiduals import UNet_hidden, UNetWide, UNetDeep, UNetDeepWide, UNetWideHidden

device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
        # specify image labels by folder name 
        self.img_labels = [item.name for item in data_dir.glob('*')]

        # construct image name list: randomly sample images for each epoch
        images = list(img_dir.glob('*' + image_type))
        self.image_name_ls = images[:2048]

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.CenterCrop([728, 728])(image)
        image = torchvision.transforms.Resize([512, 512])(image)

        # assign label to be a tensor based on the parent folder name
        label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    sample = sample.permute(0, 3, 2, 1)
    #270* rotation
    for i in range(3):
        sample = torch.rot90(sample, dims=[2, 3])
    return sample / 255.


batch_size = 1
image_size = 512
channels = 3

data_dir = pathlib.Path('../../landscapes',  fname='Combined')
train_data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
fixed_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)


def show_batch(input_batch, count=0, grayscale=False, normalize=True):
    """
    Show a batch of images with gradientxinputs superimposed

    Args:
        input_batch: arr[torch.Tensor] of input images
        output_batch: arr[torch.Tensor] of classification labels
        gradxinput_batch: arr[torch.Tensor] of atransformsributions per input image
    kwargs:
        individuals: Bool, if True then plots 1x3 image figs for each batch element
        count: int

    returns:
        None (saves .png img)

    """

    plt.figure(figsize=(15, 15))
    length, width = 1, 1
    for n in range(length*width):
        ax = plt.subplot(length, width, n+1)
        plt.axis('off')
        if normalize: 
            # rescale to [0, 1]
            input_batch[n] = (input_batch[n] - np.min(input_batch[n])) / (np.max(input_batch[n]) - np.min(input_batch[n]))
        if grayscale:
            plt.imshow(input_batch[n], cmap='gray_r')
        else:
            plt.imshow(input_batch[n])
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig('image{0:04d}.png'.format(count), dpi=300, transparent=True)
    print ('Image Saved')
    plt.close()
    return 

model = UNetDeepWide(3, 3).to(device) 
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
cosine_loss = torch.nn.CosineSimilarity(dim=0)

def train_autoencoder():
    epochs = 100
    alpha = 1
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total_mse_loss = 0
        for step, batch in enumerate(dataloader):
            lres_batch = torchvision.transforms.Resize([128, 128])(batch)
            lres_batch = torchvision.transforms.Resize([512, 512])(batch)
            lres_batch = lres_batch.to(device)
            if len(batch) < batch_size:
                break 
            optimizer.zero_grad()
            batch = batch.to(device) # discard class labels
            # batch = torchvision.transforms.Resize(random.choice(choices)*32)(batch)
            output = model(lres_batch) #, torch.ones(1).to(device))
            mse_loss = loss_fn(output, batch)
            total_mse_loss += mse_loss

            loss = (alpha * loss_fn(output, batch)) - cosine_loss(output.flatten(), batch.flatten())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
 
        print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
        print (f"Average Loss: {round(total_loss / step, 5)}")
        torch.save(model.state_dict(), 'deepwide_unet_upscale.pth')
        if (total_mse_loss / step) * alpha < 1:
            alpha *= 2

        if epoch % 5 == 0:
            batch = next(iter(fixed_dataloader)).to(device)
            gen_images = model(batch).cpu().permute(0, 2, 3, 1).detach().numpy() # torch.ones(1).to(device)
            show_batch(gen_images, count=epoch//5, grayscale=False, normalize=False)

# batch = next(iter(dataloader)).cpu().permute(0, 2, 3, 1).detach().numpy()
# model.load_state_dict(torch.load()) 
# model.eval() 
train_autoencoder()

@torch.no_grad()
def observe_denoising():
    batch = next(iter(dataloader))
    original_batch = batch
    show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=False, normalize=False)
    # alpha = 0.5
    # batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)

    original = batch[0]
    original_output = model(batch.to(device))[0][0]
    batch = torchvision.transforms.GaussianBlur(19, 8)(batch)
    transformed = batch[0]
    transformed_output = model(batch.to(device))[0][0]

    shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(shown, count=1000, grayscale=False, normalize=False)
    gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(gen_images, count=999, grayscale=False, normalize=False)
    input_distance = torch.sum((original - transformed)**2)**0.5
    output_distance = torch.sum((original_output - transformed_output)**2)**0.5
    print (f'L2 Distance on the Input after Blurring: {input_distance}')
    print (f'L2 Distance on the Autoencoder Output after Blurring: {output_distance}')

    alpha = 1
    batch = original_batch

    original = batch[0]
    original_output = model(batch.to(device))[0][0]
    batch = alpha * batch + (1-alpha) * torch.normal(0.5, 0.2, batch.shape)
    transformed = batch[0]
    transformed_output = model(batch.to(device))[0][0]

    shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(shown, count=1001, grayscale=False, normalize=False)
    gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(gen_images, count=1002, grayscale=False, normalize=False)
    input_distance = torch.sum((original - transformed)**2)**0.5
    output_distance = torch.sum((original_output - transformed_output)**2)**0.5
    print (f'L2 Distance on the Input after Gaussian Noise: {input_distance}')
    print (f'L2 Distance on the Autoencoder Output after Gaussian Noise: {output_distance}')
  
# observe_denoising()

# @torch.no_grad()
# def generate_with_noise():
#     batch = next(iter(dataloader))
#     alpha = 0
#     batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
#     for i in range(200):
#         alpha = i / 800
#         gen_images = model(batch.to(device))
#         batch = alpha * gen_images + (1-alpha) * torch.normal(0.7, 0.2, batch.shape).to(device) 
#         show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=True)

# # generate_with_noise() 

# @torch.no_grad()
# def generate_with_increasing_resolution(starting_resolution=128):
#     batch = next(iter(dataloader))
#     alpha = 0
#     batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
#     for i in range(30):
#         if i == 20:
#             batch = torchvision.transforms.Resize(256)(batch)
#         # if i == 40:
#         #     batch = torchvision.transforms.Resize(512)(batch)
#         alpha = i / 40
#         gen_images = model(batch.to(device))
#         batch = alpha * gen_images + (1-alpha) * torch.normal(0.7, 0.2, batch.shape).to(device) 
#         show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)

# generate_with_increasing_resolution()


