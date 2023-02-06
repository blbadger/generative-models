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
        self.image_name_ls = images[3072:]

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
        image = torchvision.transforms.Resize([32, 32])(image)
        # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)

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


batch_size = 16
image_size = 32
channels = 3

# path = pathlib.Path('../lsun_churches/churches/church_outdoor_train_lmdb_color_64.npy',  fname='Combined')
# dataset = npy_loader(path)
# dset = torch.utils.data.TensorDataset(dataset)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

data_dir = pathlib.Path('../nnetworks/landscapes',  fname='Combined')
train_data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# helpers
def exists(x):
	return x is not None

def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d

class Residual(nn.Module): 
	# residual connection

	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x, *args, **kwargs):
		return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
	return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
	return nn.Conv2d(dim, dim, 4, 2, 1)

class Block(nn.Module):
	def __init__(self, dim, dim_out, groups=8):
		super().__init__()
		self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
		self.norm = nn.GroupNorm(groups, dim_out)
		self.act = nn.SiLU()

	def forward(self, x, scale_shift=None):
		x = self.proj(x)
		x = self.norm(x)

		if exists(scale_shift):
			scale, shift = scale_shift
			x = x * (scale + 1) + shift

		x = self.act(x)
		return x

class ResnetBlock(nn.Module):
	"""https://arxiv.org/abs/1512.03385"""
	
	def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
		super().__init__()
		self.mlp = (
			nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
			if exists(time_emb_dim)
			else None
		)

		self.block1 = Block(dim, dim_out, groups=groups)
		self.block2 = Block(dim_out, dim_out, groups=groups)
		self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

	def forward(self, x, time_emb=None):
		h = self.block1(x)

		if exists(self.mlp) and exists(time_emb):
			time_emb = self.mlp(time_emb)
			h = rearrange(time_emb, "b c -> b c 1 1") + h

		h = self.block2(h)
		return h + self.res_conv(x)


class Attention(nn.Module):

	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**(-0.5)
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
		self.to_out = nn.Conv2d(hidden_dim, dim, 1)

	def forward(self, x):
		b, c, h, w = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = map(
			lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
		)
		q = q * self.scale

		sim = einsum("b h d i, b h d j -> b h i j", q, k)
		sim = sim - sim.amax(dim=-1, keepdim=True).detach()
		attn = sim.softmax(dim=-1)

		out = einsum("b h i j, b h d j -> b h i d", attn, v)
		out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
		return self.to_out(out)


class LinearAttention(nn.Module):

	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**-0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

		self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
									nn.GroupNorm(1, dim))

	def forward(self, x):
		b, c, h, w = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = map(
			lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
		)

		q = q.softmax(dim=-2)
		k = k.softmax(dim=-1)

		q = q * self.scale
		context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

		out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
		out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
		return self.to_out(out)


class PreNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.fn = fn
		self.norm = nn.GroupNorm(1, dim)

	def forward(self, x):
		x = self.norm(x)
		return self.fn(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return torch.randn(out.shape).to(device)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.attention = Attention(out_channels).to(device)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2 = self.attention(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 3))
        self.down1 = (Down(3, 6))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(6, 3, bilinear))
        self.outc = (OutConv(3, n_classes))
        self.attention = Attention(3).to(device)

    def forward(self, x):
        return self.attention(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        print (x2.shape)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        x = self.up4(x2, x1)
        x = self.outc(x)
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


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
    length, width = 2, 2
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

model = UNet(3, 3).to(device)
# model = UNet_hidden(3, 3).to(device)
# model = UNetWide(3, 3).to(device)
# model = UNetDeepWide(3, 3).to(device)
# model = UNetWideHidden(3, 3).to(device)
optimizer = Adam(model.parameters(), lr=1e-4) 
loss_fn = torch.nn.MSELoss()

def train_autoencoder():
    epochs = 1000
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            if len(batch) < batch_size:
                break 
            optimizer.zero_grad()
            batch = batch.to(device) # discard class labels
            output = model(batch) #, torch.ones(1).to(device))
            loss = loss_fn(output, batch) # + loss_fn(gen_res, real_res)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
        print (f"Average Loss: {round(total_loss / step, 5)}")
        torch.save(model.state_dict(), 'nonlinear_attention_64_3.pth')

        if epoch % 5 == 0:
            batch = next(iter(dataloader)).to(device)
            gen_images = model(batch).cpu().permute(0, 2, 3, 1).detach().numpy() # torch.ones(1).to(device)
            show_batch(gen_images, count=epoch//5, grayscale=False, normalize=False)

batch = next(iter(dataloader)).cpu().permute(0, 2, 3, 1).detach().numpy()
model.load_state_dict(torch.load('nonlinear_attention_64_3.pth')) 
# model.eval() 
# train_autoencoder()

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
    show_batch(shown, count=100, grayscale=False, normalize=False)
    gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(gen_images, count=99, grayscale=False, normalize=False)
    input_distance = torch.sum((original - transformed)**2)**0.5
    output_distance = torch.sum((original_output - transformed_output)**2)**0.5
    print (f'L2 Distance on the Input after Gaussian Noise: {input_distance}')
    print (f'L2 Distance on the Autoencoder Output after Gaussian Noise: {output_distance}')
  
observe_denoising()

@torch.no_grad()
def generate_with_noise():
    batch = next(iter(dataloader))
    alpha = 0
    batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
    for i in range(100):
        alpha = i / 100
        gen_images = model(batch.to(device))
        batch = alpha * gen_images + (1-alpha) * torch.normal(0.7, 0.2, batch.shape).to(device) 
        show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)

# generate_with_noise() 

@torch.no_grad()
def generate_with_increasing_resolution(starting_resolution=256):
    batch = next(iter(dataloader))
    alpha = 0
    batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
    for i in range(40):
        if i == 30:
            batch = torchvision.transforms.Resize(256)(batch)
        # if i == 30:
        #     batch = torchvision.transforms.Resize(512)(batch)
        alpha = i / 40
        gen_images = model(batch.to(device))
        batch = alpha * gen_images + (1-alpha) * torch.normal(0.7, 0.2, batch.shape).to(device) 
        show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)

# generate_with_increasing_resolution()

new_vision = model
alpha = 1
batch = next(iter(dataloader))
image = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
image = image[0].reshape(1, 3, 32, 32).to(device)
target_tensor = new_vision(image)

target_tensor = target_tensor.detach().to(device)
plt.figure(figsize=(10, 10))
image_width = len(image[0][0])
target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.imshow(target_input)
plt.axis('off')
plt.savefig('target_image', bbox_inches='tight', pad_inches=0.1)
plt.close()

modification = torch.randn(1, 3, 32, 32)/18
modification = modification.to(device)
modified_input = image + modification
modified_output = new_vision(modified_input)
print (f'L2 distance between original and shifted inputs: {torch.sqrt(torch.sum((image - modified_input)**2))}')
print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')

# plt.figure(figsize=(10, 10))
# image_width = len(modified_input[0][0])
# modified_input = modified_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
# plt.axis('off')
# plt.imshow(modified_input)
# plt.show()
# plt.close()

# alpha = 0.5
# batch = next(iter(dataloader))
# image = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
# image = image[0].reshape(1, 3, 256, 256).to(device)
# np_image = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
# plt.figure(figsize=(10, 10))
# plt.imshow(np_image)
# plt.axis('off')
# plt.savefig('a_0_image', bbox_inches='tight', pad_inches=0.1)
# plt.close()

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
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])
        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(new_vision, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
        # single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

        # if i % 500 == 0 and i > 0:
        #     output = new_vision(single_input).to(device)
        #     output_distance = torch.sqrt(torch.sum((target_tensor - output)**2))
        #     print (f'L2 distance between target and generated embedding: {output_distance}')
        #     input_distance = torch.sqrt(torch.sum((single_input - image)**2))
        #     print (f'L2 distance on the input: {input_distance}')
        #     input_distances.append(float(input_distance))
        #     output_distances.append(float(output_distance))
        #     iterations_arr.append(iterations)

    print (iterations_arr)
    print (input_distances)
    print (output_distances)
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
    dim = 32
    if random_input:
        single_input = (torch.randn(1, 3, dim, dim))/20 + 0.7 # scaled normal distribution initialization
    else:
        single_input = input_tensors

    iterations = 1000
    single_input = single_input.to(device)
    single_input = single_input.reshape(1, 3, dim, dim)
    original_input = torch.clone(single_input).reshape(3, dim, dim).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)

    single_input = octave(single_input, target_output, iterations, [0.1, 0.1], [2.4, 0.4], 0, pad=False, crop=False)

    output = model(single_input).to(device)
    print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')
    target_input = torch.tensor(target_input).reshape(1, 3, dim, dim).to(device)
    input_distance = torch.sqrt(torch.sum((single_input - image)**2))
    print (f'L2 distance on the input: {input_distance}')
    input_distances.append(float(input_distance))
    iterations_arr.append(iterations)

    print (iterations_arr)
    print (input_distances)
    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    final_input= single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy() 
    plt.axis('off')
    plt.imshow(final_input)
    plt.savefig('fig', bbox_inches='tight', pad_inches=0.1, transparent=True)
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
    loss = 0.02*torch.sum(torch.abs(target_tensor - output)) # target_tensor is the desired activation
    loss.backward()
    gradient = input_tensor.grad
    return gradient


generate_singleinput(new_vision, image, [], 0, 0, image, random_input=True)