# resnet_cifar_generator.py
# cifar10_generalization.py
# MLP-style model with GPU acceleration for latent space exploration.

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
	"""
	Creates a dataset from images classified by folder name.  
	"""

	def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
		# specify image labels by folder name 
		self.img_labels = [item.name for item in img_dir.glob('*')]
		images = list(img_dir.glob('*/*' + image_type))
		self.image_name_ls = images[:]

		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_name_ls)

	def __getitem__(self, index):
		# path to image
		img_path = os.path.join(self.image_name_ls[index])
		image = torchvision.io.read_image(img_path) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
		image = image / 255. # convert ints to floats in range [0, 1]
		image = torchvision.transforms.Resize(size=[32, 32])(image) 

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		# convert image label to tensor
		label_tens = torch.tensor(self.img_labels.index(label))
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label_tens

input_dir = pathlib.Path('CIFAR100_layer3_generated',  fname='Combined')
target_dir = pathlib.Path('CIFAR100_originals', fname='Combined')
input_trainset = ImageDataset(input_dir)
target_trainset = ImageDataset(target_dir)

batch_size = 64 # global variable
input_dataloader = torch.utils.data.DataLoader(input_trainset, batch_size=batch_size, shuffle=False)
target_dataloader = torch.utils.data.DataLoader(target_trainset, batch_size=batch_size, shuffle=False)

class FCautoencoder(nn.Module):

	def __init__(self, starting_size):

		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3, starting)
		# self.d1 = nn.Linear(starting, starting//2)
		# self.d2 = nn.Linear(starting//2, starting)
		self.d3 = nn.Linear(starting, 32*32*3)  
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(0.1) 
		self.layernorm = nn.LayerNorm(starting//4)

	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		# out = self.d1(out)
		# out = self.gelu(out)

		# out = self.d2(out)
		# out = self.gelu(out)

		out = self.d3(out)
		return out


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



def train_model(denoiser, optimizer, epochs):
	denoiser.train()
	count = 0
	total_loss = 0
	start = time.time()
	train_array, test_array = [], []

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*20)
		total_loss = 0
		count = 0
		for pair in zip(input_dataloader, target_dataloader):
			if len(pair[0][0]) < batch_size:
				break
			gen_x, target_x = pair[0][0].to(device), pair[1][0].to(device)
			gen_output = model(gen_x).reshape(batch_size, 3, 32, 32)
			loss = loss_function(gen_output, target_x)
			loss = loss.to(device)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss
			print (loss)
 
		print (f'Total Loss: {total_loss}')
		print (f"Completed in {round(time.time() - start, 2)} seconds")
		start = time.time()
		gen_output = gen_output.reshape(batch_size, 3, 32, 32).cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_output[:64, :, :, :], count=e)
		# gen_x = gen_x.reshape(batch_size, 3, 32, 32).cpu().permute(0, 2, 3, 1).cpu().detach().numpy()
		# show_batch(gen_x[:64, :, :, :], count=e)
		# target_x = target_x.reshape(batch_size, 3, 32, 32).cpu().permute(0, 2, 3, 1).cpu().detach().numpy()
		# show_batch(target_x[:64, :, :, :], count=e)
		generator_dir = 'fcnet_denoiser.pth'
		torch.save(model.state_dict(), generator_dir)

	return

loss_function = torch.nn.L1Loss()
# model = FCautoencoder(2000).to(device)
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=3, init_features=32, pretrained=False).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) 
epochs = 500
train_model(model, optimizer, epochs)

