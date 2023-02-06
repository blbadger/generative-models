import math, random, time, os
from inspect import isfunction
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
import pathlib
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms, utils
import torchvision
from torchvision.utils import save_image
from torch.optim import Adam

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import unet_noresiduals 
import unet_contractive
from prettytable import PrettyTable

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
        self.image_name_ls = images[:3072]

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
        # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)

        # assign label to be a tensor based on the parent folder name
        label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image

batch_size = 2 # global variable
image_size = 512
channels = 3

data_dir = pathlib.Path('../nnetworks/landscapes',  fname='Combined')
train_data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
print (len(dataloader)) 

# transform = transforms.Compose([
#           transforms.Resize((image_size, image_size)),
#           transforms.ToTensor()
# ])

# def npy_loader(path):
#   sample = torch.from_numpy(np.load(path))
#   sample = sample.permute(0, 3, 2, 1)
#   #270* rotation
#   for i in range(3):
#       sample = torch.rot90(sample, dims=[2, 3])
#   return sample / 255.

# path = pathlib.Path('../nnetworks/lsun_churches/churches/church_outdoor_train_lmdb_color_64.npy',  fname='Combined')

# dataset = npy_loader(path)
# dset = torch.utils.data.TensorDataset(dataset)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

class SmallFCEncoder(nn.Module):

    def __init__(self, starting_size, channels):
        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(64*64*3, starting)
        self.d1 = nn.Linear(starting, starting//4)
        self.d2 = nn.Linear(starting//4, starting//8)
        self.d3 = nn.Linear(starting//8, starting//4)
        self.d4 = nn.Linear(starting//4, starting)
        self.d5 = nn.Linear(starting, 64*64*3)
        self.gelu = nn.GELU()
        self.layernorm1 = nn.LayerNorm(starting)
        self.layernorm2 = nn.LayerNorm(starting//2)
        self.layernorm3 = nn.LayerNorm(starting//4)
        self.layernorm4 = nn.LayerNorm(starting//2)
        self.layernorm5 = nn.LayerNorm(starting)

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        
        out = self.input_transform(input_tensor)
        out = self.layernorm1(self.gelu(out))

        out = self.d1(out)
        out = self.gelu(out)

        out = self.d2(out)
        out = self.gelu(out)

        out = self.d3(out)
        out = self.gelu(out)

        out = self.d4(out)
        out = self.gelu(out)

        out = self.d5(out)
        out = out.reshape(batch_size, channels, image_size, image_size)
        return out

class FCEncoder(nn.Module):

    def __init__(self, starting_size, channels):
        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(64*64*3, starting)
        self.d1 = nn.Linear(starting, starting)
        self.d2 = nn.Linear(starting, starting)
        self.d3 = nn.Linear(starting, starting)
        self.d4 = nn.Linear(starting, starting)
        self.d5 = nn.Linear(starting, 64*64*3)
        self.layernorm1 = nn.LayerNorm(starting)
        self.gelu = nn.GELU()

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        
        out = self.input_transform(input_tensor)
        out = self.layernorm1(self.gelu(out))

        out1 = self.d1(out)
        out = self.gelu(out1) 

        out2 = self.d2(out)
        out = self.gelu(out2)

        out = self.d3(out)
        out = self.gelu(out)

        out = self.d4(out)
        out = self.gelu(out)

        out = self.d5(out)
        out = out.reshape(batch_size, channels, image_size, image_size)
        return out 



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

    plt.figure(figsize=(8, 5))
    length, width = 1, 2
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

# model = FCEncoder(10000, 1).to(device) 
# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=3, init_features=32, pretrained=False).to(device)

model = unet_noresiduals.UNet(n_channels=3, n_classes=3).to(device)

batch = next(iter(dataloader)).cpu().permute(0, 2, 3, 1).detach().numpy()
model.load_state_dict(torch.load('unet_autoencoder_landscapes_512.pth'))
model.eval()

batch = next(iter(dataloader))
show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=False, normalize=False)
alpha = 1
batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
print (batch.shape) 
shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
show_batch(shown, count=100, grayscale=False, normalize=False)
gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
show_batch(gen_images, count=99, grayscale=False, normalize=False)

# batch = next(iter(dataloader))
# show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=False, normalize=False)
# alpha = 0.1
# batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
# print (batch.shape)
# shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
# show_batch(shown, count=100, grayscale=False, normalize=False)
# gen_images = model(batch.to(device))[0].cpu().permute(0, 2, 3, 1).detach().numpy()
# show_batch(gen_images, count=99, grayscale=False, normalize=False)


def random_crop(input_image, size):
    """
    Crop an image with a starting x, y coord from a uniform distribution

    Args:
        input_image: torch.tensor object to be cropped
        size: int, size of the desired image (size = length = width)

    Returns:
        input_image_cropped: torch.tensor
        crop_height: starting y coordinate
        crop_width: starting x coordinate
    """

    image_width = len(input_image[0][0])
    image_height = len(input_image[0])
    crop_width = random.randint(0, image_width - size)
    crop_height = random.randint(0, image_width - size)
    input_image_cropped = input_image[:, :, crop_height:crop_height + size, crop_width: crop_width + size]

    return input_image_cropped, crop_height, crop_width

def octave(single_input, target_output, iterations, learning_rates, sigmas, size, pad=False, crop=True):
    """
    Perform an octave (scaled) gradient descent on the input.

    Args;
        single_input: torch.tensor of the input
        target_output: torch.tensor of the desired output category
        iterations: int, the number of iterations desired
        learning_rates: arr[int, int], pair of integers corresponding to start and end learning rates
        sigmas: arr[int, int], pair of integers corresponding to the start and end Gaussian blur sigmas
        size: int, desired dimension of output image (size = length = width)

    kwargs:
        pad: bool, if True then padding is applied at each iteration of the octave
        crop: bool, if True then gradient descent is applied to cropped sections of the input

    Returns:
        single_input: torch.tensor of the transformed input
    """

    start_lr, end_lr = learning_rates
    start_sigma, end_sigma = sigmas
    iterations_arr, input_distances, output_distances = [], [], []
    for i in range(iterations):
        print (i)
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])
        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(model, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad # gradient descent step
        # single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

    return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, target_input, random_input=True):
    """
    Generates an input for a given output

    Args:
        input_tensor: torch.Tensor object, minibatch of inputs
        output_tensor: torch.Tensor object, minibatch of outputs
        index: int, target class index to generate
        cout: int, time step

    kwargs: 
        random_input: bool, if True then a scaled random normal distributionis used

    returns:
        None (saves .png image)
    """

    # manualSeed = 999
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    class_index = index
 
    input_distances = []
    iterations_arr = []
    if random_input:
        single_input = (torch.randn(1, 3, 512, 512))/20 + 0.7 # scaled normal distribution initialization

    else:
        single_input = input_tensors[0]

    iterations = 500
    single_input = single_input.to(device)
    single_input = single_input.reshape(1, 3, 512, 512)
    original_input = torch.clone(single_input).reshape(3, 512, 512).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)
    single_input = octave(single_input, target_output, iterations, [0.1, 0.1], [2.4, 0.4], 0, pad=False, crop=False)

    output = model(single_input).to(device)
    print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')
    target_input = torch.tensor(target_input).reshape(1, 3, 512, 512).to(device)
    input_distance = torch.sqrt(torch.sum((single_input - image)**2))
    print (f'L2 distance on the input: {input_distance}')
    input_distances.append(float(input_distance))
    iterations_arr.append(iterations)

    print (iterations_arr)
    print (input_distances)
    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    generated_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(generated_input)
    plt.savefig('fig', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return 


def layer_gradient(model, input_tensor, desired_output):
    """
    Compute the gradient of the output (logits) with respect to the input 
    using an L1 metric to maximize the target classification.

    Args:
        model: torch.nn.model
        input_tensor: torch.tensor object corresponding to the input image
        true_output: torch.tensor object of the desired classification label

    Returns:
        gradient: torch.tensor.grad on the input tensor after backpropegation

    """
    input_tensor.requires_grad = True
    output = model(input_tensor)
    loss = 0.005*torch.sum(torch.abs(target_tensor - output)) # target_tensor is the desired activation
    loss.backward()
    gradient = input_tensor.grad
    return gradient

image = next(iter(dataloader))[0].reshape(1, 3, 512, 512).to(device)
with torch.no_grad():
    target_tensor = model(image)
    modification = torch.randn(1, 3, 512, 512)/18
    modification = modification.to(device)
    modified_input = image + modification
    modified_output = model(modified_input)
    print (f'L2 distance between original and shifted inputs: {torch.sqrt(torch.sum((image - modified_input)**2))}')
    print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')
    
generate_singleinput(model, [], [], 0, 0, image)

