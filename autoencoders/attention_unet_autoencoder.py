import math
from inspect import isfunction
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
import time
import pathlib
import numpy as np
import os

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
		return self.fn(x, *args, **kwargs) + (0.05 * x + 0.95 * torch.normal(0.5, 0.2, x.shape).to(device ))


def Upsample(dim):
	return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
	return nn.Conv2d(dim, dim, 4, 2, 1)


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
		return h + (0.05*self.res_conv(x) + 0.95 * torch.normal(0.5, 0.2, h.shape).to(device))

class Attention(nn.Module):

	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**-0.5
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

class PositionalEmbeddings(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim


	def forward(self, time):
		device = time.device
		half_dim = self.dim // 2
		embeddings = math.log(1000) / (half_dim - 1)
		embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
		embeddings = time[:, None] * embeddings[None, :]
		embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
		return embeddings

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



class Unet(nn.Module):
	def __init__(
		self,
		dim,
		init_dim=None,
		out_dim=None,
		dim_mults=(1, 2, 4, 8),
		channels=3,
		with_time_emb=False,
		resnet_block_groups=8,
		use_convnext=False,
		convnext_mult=2,
	):
		super().__init__()

		# determine dimensions
		self.channels = channels

		init_dim = default(init_dim, dim // 3 * 2)
		self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

		dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
		in_out = list(zip(dims[:-1], dims[1:]))
		
		block_klass = partial(ResnetBlock, groups=resnet_block_groups)

		# time embeddings
		if with_time_emb:
			time_dim = dim * 4
			self.time_mlp = nn.Sequential(
				PositionalEmbeddings(dim),
				nn.Linear(dim, time_dim),
				nn.GELU(),
				nn.Linear(time_dim, time_dim),
			)
		else:
			time_dim = None
			self.time_mlp = None

		# layers
		self.downs = nn.ModuleList([])
		self.ups = nn.ModuleList([])
		num_resolutions = len(in_out)

		for ind, (dim_in, dim_out) in enumerate(in_out):
			is_last = ind >= (num_resolutions - 1)

			self.downs.append(
				nn.ModuleList(
					[
						block_klass(dim_in, dim_out, time_emb_dim=time_dim),
						block_klass(dim_out, dim_out, time_emb_dim=time_dim),
						Residual(PreNorm(dim_out, LinearAttention(dim_out))),
						Downsample(dim_out) if not is_last else nn.Identity(),
					]
				)
			)

		mid_dim = dims[-1]
		self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
		self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
		self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
			is_last = ind >= (num_resolutions - 1)

			self.ups.append(
				nn.ModuleList(
					[
						block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
						block_klass(dim_in, dim_in, time_emb_dim=time_dim),
						Residual(PreNorm(dim_in, LinearAttention(dim_in))),
						Upsample(dim_in) if not is_last else nn.Identity(),
					]
				)
			)

		out_dim = default(out_dim, channels)
		self.final_conv = nn.Sequential(
			block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
		)

	def forward(self, x, time):
		x = self.init_conv(x)

		t = self.time_mlp(time) if exists(self.time_mlp) else None

		h = []

		# for block1, block2, downsample in self.downs:
		# 	x = block1(x, t)
		# 	x = block2(x, t)
		# 	h.append(x)
		# 	x = downsample(x)

		# downsample
		for block1, block2, attn, downsample in self.downs:
			x = block1(x, t)
			x = block2(x, t)
			x = attn(x)
			h.append(x)
			x = downsample(x)

		# bottleneck
		x = self.mid_block1(x, t)
		x = self.mid_attn(x)
		x = self.mid_block2(x, t)

		# for block1, block2, upsample in self.ups:
		# 	x = torch.cat((x, x), dim=1)
		# 	x = block1(x, t)
		# 	x = block2(x, t)
		# 	x = upsample(x)

		# upsample
		for block1, block2, attn, upsample in self.ups:
			x = torch.cat((x, h.pop()), dim=1)
			x = block1(x, t)
			x = block2(x, t)
			x = attn(x)
			x = upsample(x)

		return self.final_conv(x)


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
		image = torchvision.transforms.Resize([128, 128])(image)
		# image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image

batch_size = 16
image_size = 128
channels = 3

data_dir = pathlib.Path('../nnetworks/landscapes',  fname='Combined')
train_data = ImageDataset(data_dir, image_type='.jpg')
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

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
	length, width = 4, 4
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

def count_parameters(model):
    """
    Display the tunable parameters in the model of interest

    Args:
        model: torch.nn object

    Returns:
        total_params: the number of model parameters

    """

    table = PrettyTable(['Module', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param 	 

    print (table)
    print (f'Total trainable parameters: {total_params}')
    return total_params

model = Unet(
	dim=image_size,
	channels=channels,
	dim_mults=(1, 2, 4)
).to(device)

# model = unet_noresiduals.UNet_hidden(3, 3).to(device)

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
			output = model(batch, torch.ones(1).to(device)) #, torch.ones(1).to(device))
			loss = loss_fn(output, batch) # + loss_fn(gen_res, real_res)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()

		print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
		print (f"Average Loss: {round(total_loss / step, 5)}")
		torch.save(model.state_dict(), 'unetattention_fclandscapes_128.pth')

		if epoch % 5 == 0:
			batch = next(iter(dataloader)).to(device)
			gen_images = model(batch, torch.ones(1).to(device)).cpu().permute(0, 2, 3, 1).detach().numpy() # torch.ones(1).to(device)
			show_batch(gen_images, count=epoch//5, grayscale=False, normalize=False)

batch = next(iter(dataloader)).cpu().permute(0, 2, 3, 1).detach().numpy()
model.load_state_dict(torch.load('unetattention_fclandscapes_128.pth')) 
# train_autoencoder()

@torch.no_grad()
def random_manifold_walk():
	data = iter(dataloader)
	batch = next(data).to(device)
	output, hidden = model(batch)
	gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=0, grayscale=False, normalize=True)

	unet_decoder = UnetDecoder(model, batch)
	og_hidden = hidden
	for i in range(30):
		random = torch.normal(0, 0.5, hidden.shape).to(device)
		hidden += random 
		output = unet_decoder(hidden)
		gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i + 1, grayscale=False, normalize=True)
		out, hidden = model(output)
	return

@torch.no_grad()
def directed_manifold_walk():
	data = iter(dataloader)
	batch = next(data).to(device)
	batch2 = next(data).to(device)
	output, hidden_original = model(batch)
	target_output, target_hidden = model(batch2)
	gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=0, grayscale=False, normalize=True)
	unet_decoder = UnetHiddenDecoder(model, batch)
	for i in range(60):
		alpha = i/60
		hidden = (1- alpha) * hidden_original + alpha * target_hidden
		output = unet_decoder(hidden)
		gen_images = output[0].cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i + 1, grayscale=False, normalize=True)

	return

@torch.no_grad()
def observe_denoising():
	batch = next(iter(dataloader))
	original_batch = batch
	show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=False, normalize=False)
	# batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)

	# original = batch[0]
	# original_output = model(batch.to(device))[0][0]
	# batch = torchvision.transforms.GaussianBlur(19, 8)(batch)
	# transformed = batch[0]
	# transformed_output = model(batch.to(device))[0][0]

	# shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(shown, count=1000, grayscale=False, normalize=False)
	# gen_images = model(batch.to(device))[0].cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(gen_images, count=999, grayscale=False, normalize=False)
	# input_distance = torch.sum((original - transformed)**2)**0.5
	# output_distance = torch.sum((original_output - transformed_output)**2)**0.5
	# print (f'L2 Distance on the Input after Blurring: {input_distance}')
	# print (f'L2 Distance on the Autoencoder Output after Blurring: {output_distance}')

	alpha = 0.5
	batch = original_batch

	original = batch[0]
	original_output = model(batch.to(device), torch.ones(1).to(device))[0]
	batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
	transformed = batch[0]
	transformed_output = model(batch.to(device), torch.ones(1).to(device))[0]

	shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(shown, count=100, grayscale=False, normalize=False)
	gen_images = model(batch.to(device), torch.ones(1).to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=99, grayscale=False, normalize=False)
	input_distance = torch.sum((original - transformed)**2)**0.5
	output_distance = torch.sum((original_output - transformed_output)**2)**0.5
	print (f'L2 Distance on the Input after Gaussian Noise: {input_distance}')
	print (f'L2 Distance on the Autoencoder Output after Gaussian Noise: {output_distance}')

observe_denoising()
# model.eval()
# random_manifold_walk()

@torch.no_grad()
def generate_with_noise():
	batch = next(iter(dataloader))
	alpha = 0
	batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
	for i in range(50):
		alpha = i / 50
		gen_images = model(batch.to(device))
		show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)
		batch = alpha * gen_images + (1-alpha) * torch.normal(0.5, 0.2, batch.shape).to(device) 

# generate_with_noise()
