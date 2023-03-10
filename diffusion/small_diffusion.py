import math, random, time, os
from inspect import isfunction
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

batch_size = 128 # global variable
image_size = 128
channels = 3
timesteps = 1000

transform = transforms.Compose([
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Lambda(lambda t: (t * 2) - 1)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

class FCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3 + 1, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting//4)
		self.d3 = nn.Linear(starting//4, starting//2)
		self.d4 = nn.Linear(starting//2, starting)
		self.d5 = nn.Linear(starting, 32*32*3)
		self.gelu = nn.GELU()
		self.layernorm1 = nn.LayerNorm(starting)
		self.layernorm2 = nn.LayerNorm(starting//2)
		self.layernorm3 = nn.LayerNorm(starting//4)
		self.layernorm4 = nn.LayerNorm(starting//2)
		self.layernorm5 = nn.LayerNorm(starting)

	def forward(self, input_tensor, time):
		time_tensor = torch.tensor(time/timesteps).reshape(batch_size, 1)
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		
		input_tensor = torch.cat((input_tensor, time_tensor), dim=-1)
		out0 = self.input_transform(input_tensor)
		out = self.layernorm1(self.gelu(out0))

		out1 = self.d1(out)
		out = self.layernorm2(self.gelu(out1))

		out2 = self.d2(out)
		out = self.layernorm3(self.gelu(out2)) + out2

		out = self.d3(out)
		out = self.layernorm4(self.gelu(out)) + out1

		out = self.d4(out)
		out = self.layernorm5(self.gelu(out)) + out0

		out = self.d5(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out


class BiggerFCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(image_size*image_size*channels + 1, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting//4)
		self.d3 = nn.Linear(starting//4, starting//4)
		self.d4 = nn.Linear(starting//4, starting//8)
		self.d5 = nn.Linear(starting//8, starting//8)
		self.d6 = nn.Linear(starting//8, starting//4)
		self.d7 = nn.Linear(starting//4, starting//4)
		self.d8 = nn.Linear(starting//4, starting//2)
		self.d9 = nn.Linear(starting//2, starting)
		self.d10 = nn.Linear(starting, image_size*image_size*channels)
		self.gelu = nn.GELU()

		self.layernorm1 = nn.LayerNorm(starting)
		self.layernorm2 = nn.LayerNorm(starting//2)
		self.layernorm3 = nn.LayerNorm(starting//4)
		self.layernorm4 = nn.LayerNorm(starting//4)
		self.layernorm5 = nn.LayerNorm(starting//8)
		self.layernorm6 = nn.LayerNorm(starting//8)
		self.layernorm7 = nn.LayerNorm(starting//4)
		self.layernorm8 = nn.LayerNorm(starting//4)
		self.layernorm9 = nn.LayerNorm(starting//2)
		self.layernorm10 = nn.LayerNorm(starting)

	def forward(self, input_tensor, time):
		time_tensor = torch.tensor(time/timesteps).reshape(batch_size, 1)
		input_tensor = torch.flatten(input_tensor, start_dim=1)
	
		input_tensor = torch.cat((input_tensor, time_tensor), dim=-1)

		out0 = self.input_transform(input_tensor)
		out = self.layernorm1(self.gelu(out0))

		out1 = self.d1(out)
		out = self.layernorm2(self.gelu(out1))

		out2 = self.d2(out)
		out = self.layernorm3(self.gelu(out2))

		out3 = self.d3(out)
		out = self.layernorm4(self.gelu(out3))

		out4 = self.d4(out)
		out = self.layernorm5(self.gelu(out4))

		out = self.d5(out)
		out = self.layernorm6(self.gelu(out)) + out4

		out = self.d6(out)
		out = self.layernorm7(self.gelu(out)) + out3

		out = self.d7(out)
		out = self.layernorm8(self.gelu(out)) + out2

		out = self.d8(out)
		out = self.layernorm9(self.gelu(out)) + out1

		out = self.d9(out)
		out = self.layernorm10(self.gelu(out)) + out0

		out = self.d10(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out


def linear_beta_schedule(timesteps):
	beta_start = 0.0001
	beta_end = 0.02
	return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
	"""
	cosine schedule as proposed in https://arxiv.org/abs/2102.09672
	"""
	steps = timesteps + 1
	x = torch.linspace(0, timesteps, steps)
	alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clip(betas, 0.0001, 0.9999)

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
	batch_size = t.shape[0]
	out = a.gather(-1, t.cpu())
	return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
	if noise is None:
		noise = torch.randn_like(x_start)

	x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
	predicted_noise = denoise_model(x_noisy, t)
	if loss_type == 'random':
		losses = ['l1', 'l2', 'huber']
		loss_type = random.choice(losses)

	if loss_type == 'l1':
		loss = F.l1_loss(noise, predicted_noise)
	elif loss_type == 'l2':
		loss = F.mse_loss(noise, predicted_noise)
	elif loss_type == "huber":
		loss = F.smooth_l1_loss(noise, predicted_noise)
	elif loss_type == 'uniform':
		loss = torch.max(F.l1_loss(noise, predicted_noise))
	else:
		raise NotImplementedError()

	return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
	betas_t = extract(betas, t, x.shape)
	sqrt_one_minus_alphas_cumprod_t = extract(
		sqrt_one_minus_alphas_cumprod, t, x.shape
	)
	sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
	
	# Equation 11 in the paper
	# Use our model (noise predictor) to predict the mean
	model_mean = sqrt_recip_alphas_t * (
		x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
	)

	if t_index == 0:
		return model_mean
	else:
		posterior_variance_t = extract(posterior_variance, t, x.shape)
		noise = torch.randn_like(x)
		# Algorithm 2 line 4:
		return model_mean + torch.sqrt(posterior_variance_t) * noise 


def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.permute(0, 2, 3, 1).cpu().numpy())
    return imgs


def sample(model, image_size, batch_size=16, channels=3):
	return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
	if noise is None:
		noise = torch.randn_like(x_start)

	sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
	sqrt_one_minus_alphas_cumprod_t = extract(
		sqrt_one_minus_alphas_cumprod, t, x_start.shape
	)
	return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def num_to_groups(num, divisor):
	groups = num // divisor
	remainder = num % divisor
	arr = [divisor] * groups
	if remainder > 0:
		arr.append(remainder)
	return arr


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
	length, width = 8, 8
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
	plt.savefig('image{0:04d}.png'.format(count), dpi=300)
	print ('Image Saved')
	plt.close()
	return 

model = FCEncoder(10000, 1).to(device) # 10000
optimizer = Adam(model.parameters(), lr=1e-4) # 1e-5
latent_noise = torch.randn(batch_size, channels, image_size, image_size, device=device)

def train_diffusion_inversion():
	epochs = 1000
	for epoch in range(epochs):
		start_time = time.time()
		total_loss = 0
		for step, batch in enumerate(dataloader):
			batch = batch[0]
			if len(batch) < batch_size:
				break 
			optimizer.zero_grad()
			batch = batch.to(device) # discard class labels
			timestep = torch.randint(0, timesteps, (batch_size,), device=device).long().to(device) # integer
			loss = p_losses(model, batch, timestep, loss_type='l2')
			total_loss += loss.item()
	 
			loss.backward()
			optimizer.step()
		print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
		print (f"Average Loss: {round(total_loss / step, 5)}")
		torch.save(model.state_dict(), 'fcnet_cifar_diffusion_test.pth')

		if epoch % 5 == 0:
			gen_images = sample(model, image_size, batch_size=batch_size, channels=channels)
			gen_images = (gen_images[-1] + 1) / 2
			show_batch(gen_images, count=epoch//5, grayscale=False, normalize=True)

model.load_state_dict(torch.load('fcnet_cifar_diffusion_test.pth'))
train_diffusion_inversion()