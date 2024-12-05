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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from undercomplete_autoencoder import count_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

batch_size = 128 # global variable
image_size = 64
channels = 3
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# define image transformations (e.g. using torchvision)
transform = transforms.Compose([
			# transforms.RandomHorizontalFlip(),
			transforms.Resize((64, 64)),
			transforms.ToTensor(),
			transforms.Lambda(lambda t: (t * 2) - 1)
])

def train_autoencoder(model, dataset='churches'):
    epochs = 500
    alpha = 1
    gpu_count = torch.cuda.device_count()
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if dataset == 'churches':
        path = pathlib.Path('/home/bbadger/Desktop/church_outdoor_train_lmdb_color_64.npy', fname='Combined')
        dataset = npy_loader(path)
        dset = torch.utils.data.TensorDataset(dataset)
        start = (len(dset) // gpu_count) * rank
        end = start + (len(dset) // gpu_count)
        dataloader = torch.utils.data.DataLoader(dataset[start:end], batch_size=batch_size, shuffle=True)

    # landscapes dataset load
    else:
        data_dir = pathlib.Path('/home/bbadger/Desktop/landscapes', fname='Combined')
        train_data = ImageDataset(data_dir, rank, image_type='.jpg')
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # create model and move it to GPU with id rank

    device_id = rank % torch.cuda.device_count()
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    optimizer = Adam(ddp_model.parameters(), lr=1e-4)
    ddp_model.train()

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        total_loss = 0
        total_mse_loss = 0

        for step, batch in enumerate(dataloader):
            if len(batch) < batch_size:
                break
            optimizer.zero_grad()
            batch = batch.to(device_id) # discard class labels
            output = ddp_model(batch)
  #          output = torch.clip(output, min=0, max=1)
            loss = loss_fn(output, batch)
            loss = torch.masked_select(loss, loss > 0.01)
            loss_size = loss.shape
            loss = torch.mean(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if rank == 0:
            checkpoint_path = f'/home/bbadger/Desktop/churches_unetdeepwide/{epoch}'
            if epoch % 4 == 0: torch.save(ddp_model.state_dict(), checkpoint_path)
            tqdm.write(f"Epoch {epoch} completed in {time.time() - start_time} seconds")
            tqdm.write(f"Average Loss: {round(total_loss / step, 5)}")
            tqdm.write(f"Loss shape: {loss_size}")
        dist.barrier()

    dist.destroy_process_group()

if __name__ == '__main__':
    train_autoencoder(model, dataset='churches')
t
def npy_loader(path):
	sample = torch.from_numpy(np.load(path))
	sample = sample.permute(0, 3, 2, 1)
	#270* rotation
	for i in range(3):
		sample = torch.rot90(sample, dims=[2, 3])
	return sample / 255.

# path = pathlib.Path('../nnetworks/lsun_churches/churches/church_outdoor_train_lmdb_color_64.npy',  fname='Combined')

# dataset = npy_loader(path)
# dset = torch.utils.data.TensorDataset(dataset)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

class ConvNextBlock(nn.Module):
	"""https://arxiv.org/abs/2201.03545"""

	def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
		super().__init__()
		self.mlp = (
			nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
			if exists(time_emb_dim)
			else None
		)

		self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

		self.net = nn.Sequential(
			nn.GroupNorm(1, dim) if norm else nn.Identity(),
			nn.Conv2d(dim, dim_out * mult, 3, padding=1),
			nn.GELU(),
			nn.GroupNorm(1, dim_out * mult),
			nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
		)

		self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

	def forward(self, x, time_emb=None):
		h = self.ds_conv(x)

		if exists(self.mlp) and exists(time_emb):
			condition = self.mlp(time_emb)
			h = h + rearrange(condition, "b c -> b c 1 1")

		h = self.net(h)
		return h + self.res_conv(x)


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


class Unet(nn.Module):
	def __init__(
		self,
		dim,
		init_dim=None,
		out_dim=None,
		dim_mults=(1, 2, 4, 8),
		channels=3,
		with_time_emb=True,
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

		# upsample
		for block1, block2, attn, upsample in self.ups:
			x = torch.cat((x, h.pop()), dim=1)
			x = block1(x, t)
			x = block2(x, t)
			x = attn(x)
			x = upsample(x)

		return self.final_conv(x)

class FCautoencoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*channels, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting//4)
		self.d3 = nn.Linear(starting//4, starting//2)
		self.d4 = nn.Linear(starting//2, starting)
		self.d5 = nn.Linear(starting, 32*32*channels)
		self.gelu = nn.GELU()
		self.layernorm1 = nn.LayerNorm(starting)
		self.layernorm2 = nn.LayerNorm(starting//2)
		self.layernorm3 = nn.LayerNorm(starting//4)
		self.layernorm4 = nn.LayerNorm(starting//2)

	def forward(self, input_tensor, time):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		out1 = self.d1(out)
		out = self.gelu(out1)

		out2 = self.d2(out)
		out = self.gelu(out2)

		out = self.d3(out)
		out = self.gelu(out) + out2

		out = self.d4(out)
		out = self.gelu(out) + out1

		out = self.d5(out)
		return out


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

def linear_beta_schedule(timesteps):
	beta_start = 0.0001
	beta_end = 0.02
	return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
	beta_start = 0.0001
	beta_end = 0.02
	return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
	beta_start = 0.0001
	beta_end = 0.02
	betas = torch.linspace(-6, 6, timesteps)
	return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 1000

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

	if loss_type == 'l1':
		loss = F.l1_loss(noise, predicted_noise)
	elif loss_type == 'l2':
		loss = F.mse_loss(noise, predicted_noise)
	elif loss_type == "huber":
		loss = F.smooth_l1_loss(noise, predicted_noise)
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


def show_batch(input_batch, count=0, grayscale=False):
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
	for n in range(min(batch_size, 64)):
		ax = plt.subplot(8, 8, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray_r')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.savefig('image{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

model = Unet(
	dim=image_size,
	channels=channels,
	dim_mults=(1, 2, 4)
).to(device)

# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=1, out_channels=1, init_features=32, pretrained=True).to(device)

# model = FCEncoder().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
print (model)

epochs = 10
for epoch in range(epochs):
	start_time = time.time()
	total_loss = 0
	for step, batch in enumerate(dataloader):
		 
 		if len(batch) < batch_size:
 			break
 		optimizer.zero_grad()
 		batch = batch.to(device) # discard class labels
 		timestep = torch.randint(0, timesteps, (batch_size,), device=device).long().to(device) # integer
 		loss = p_losses(model, batch, timestep, loss_type="l2")
 		total_loss += loss.item()
 
 		loss.backward()
 		optimizer.step()
 	print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
 	print (f"Average Loss: {round(total_loss / step, 5)}")

count_parameters(model)
n = 64
# gen_images = sample(model, 64, batch_size=n, channels=channels)
# gen_images = gen_images.cpu().numpy()
# show_batch((gen_images[-1] + 1) / 2, grayscale=True)

