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
from prettytable import PrettyTable

device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

batch_size = 256 # global variable
image_size = 32
channels = 3
timesteps = 1000

transform = transforms.Compose([
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor()
])

def npy_loader(path):
	sample = torch.from_numpy(np.load(path))
	sample = sample.permute(0, 3, 2, 1)
	#270* rotation
	for i in range(3):
		sample = torch.rot90(sample, dims=[2, 3])
	return sample / 255.

path = pathlib.Path('../nnetworks/lsun_churches/churches/church_outdoor_train_lmdb_color_64.npy',  fname='Combined')

# dataset = npy_loader(path)
# dset = torch.utils.data.TensorDataset(dataset)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

class NewResnet(nn.Module):

	def __init__(self, model, n_output):
		super().__init__()
		self.model = model
		self.inplanes = 64
		self.fc = nn.Linear(512, n_output) # 512 * 4 for resnet50 or resnet152
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=2, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x = self.model.conv1(x)
		x = self.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)

		x = self.model.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		x = self.sigmoid(x)
		return x

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

class FCnet(nn.Module):

	def __init__(self):

		super().__init__()
		self.input_transform = nn.Linear(32*32*3, 1000)
		self.d1 = nn.Linear(1000, 600)
		self.d2 = nn.Linear(600, 200)
		self.d3 = nn.Linear(200, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.)

	def forward(self, input_tensor):
		out = self.input_transform(input_tensor)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d1(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d2(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.d3(out)
		out = self.sigmoid(out)
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

@torch.no_grad()
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

autoencoder = unet_noresiduals.UNet(n_channels=3, n_classes=3).to(device)
discriminator = NewResnet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True), 1).to(device)
count_parameters(autoencoder)

discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-4)
autoencoder_optimizer = Adam(autoencoder.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()
adversarial_loss = torch.nn.BCELoss()

def train_autoencoder():
	epochs = 100
	for epoch in range(epochs):
		start_time = time.time()
		total_loss = 0
		for step, batch in enumerate(dataloader):
			print (step)
			autoencoder_optimizer.zero_grad()
			discriminator_optimizer.zero_grad()
			batch = batch[0]
			if len(batch) < batch_size:
				break 
			batch = batch.to(device) # discard class labels 
			output = autoencoder(batch)
			autoencoder_loss = loss_fn(output, batch)
			total_loss += autoencoder_loss.item() 
			autoencoder_loss.backward(retain_graph=True)

			discriminator_outputs = discriminator(output).reshape(batch_size)
			autoencoder_loss = adversarial_loss(discriminator_outputs, torch.ones(batch_size).to(device)) # pretend that all generated inputs are real
			autoencoder_loss.backward(retain_graph=True)

			first_loss = adversarial_loss(discriminator(batch).reshape(batch_size), torch.ones(batch_size).to(device))
			discriminator_loss = adversarial_loss(discriminator_outputs, torch.zeros(batch_size).to(device)) + first_loss
			discriminator_loss.backward()
			discriminator_optimizer.step()
			autoencoder_optimizer.step()

		print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
		print (f"Average Loss: {round(total_loss / step, 5)}")
		torch.save(autoencoder.state_dict(), 'unet_adversarial_autoencoder_cifar.pth')

		if epoch % 1 == 0:
			batch = next(iter(dataloader))[0].to(device)
			gen_images = autoencoder(batch).cpu().permute(0, 2, 3, 1).detach().numpy()
			show_batch(gen_images, count=epoch, grayscale=False, normalize=True)

autoencoder.load_state_dict(torch.load('unet_adversarial_autoencoder_cifar.pth'))
train_autoencoder()

def interpolate_latent():
	data = iter(dataloader)
	batch1 = next(data)[0].to(device)
	batch2 = next(data)[0].to(device)
	random = torch.normal(0.5, 0.2, batch1.shape).to(device)
	for i in range(31):
		alpha = 1 - i / 30
		# if i <= 30:
		# 	beta = i / 30
		# else:
		# 	beta = abs(2 - i / 30)
		batch = alpha * batch1 + (1 - alpha)* batch2 #+ 2 * beta * random 
		gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i, grayscale=False, normalize=True)

	return

# interpolate_latent()

# batch = next(iter(dataloader))[0]
# show_batch(torchvision.transforms.Resize((32, 32))(batch).cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=False, normalize=False)
# alpha = 0.9
# batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
# print (batch.shape)
# shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
# show_batch(shown, count=100, grayscale=False, normalize=False)
# gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
# show_batch(gen_images, count=99, grayscale=False, normalize=False)









