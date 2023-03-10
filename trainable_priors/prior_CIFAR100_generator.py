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

transform = transforms.Compose(
	[transforms.ToTensor()])

# stats = ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
# train_transforms= transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
# 						 transforms.RandomHorizontalFlip(), 
# 						 transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.ToTensor()])

batch_size = 64 # global variable
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size, shuffle=False)
# testloader = torch.utils.data.DataLoader(new_testset, batch_size=batch_size, shuffle=False)

class NewResNet(nn.Module):

	def __init__(self, model, num_classes):
		super().__init__()
		self.model = model
		self.inplanes = 64
		self.fc = nn.Linear(512 * 4, num_classes)
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=2, bias=False)

	def forward(self, x):
		x = self.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		# x = self.model.layer4(x)

		# x = self.model.avgpool(x)
		# x = torch.flatten(x, 1)
		# x = self.fc(x)
		return x


class FCautoencoder(nn.Module):

	def __init__(self, starting_size):

		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting)
		self.d3 = nn.Linear(starting, 32*32*3)
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(0.1) 
		self.layernorm = nn.LayerNorm(starting//4)

	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		out = self.d1(out)
		out = self.gelu(out)

		out = self.d2(out)
		out = self.gelu(out)

		out = self.d3(out)
		return out

def random_crop(input_image, size):
	"""
	Crop an image with a starting x, y coord from a uniform distribution
`
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

gen_dir_counts = {i:0 for i in range(100)}
orig_dir_counts = {i:0 for i in range(100)}
def save_images(input_batch, output_batch, gen_dir_counts=gen_dir_counts, orig_dir_counts=orig_dir_counts, batch_size=batch_size, generated=True):
	if generated:
		data_dir = 'CIFAR100_layer3_generated'
	else:
		data_dir = 'CIFAR100_layer3_originals'

	for i in range(batch_size):
		print (i)
		image = input_batch[i, :, :, :]
		category = int(output_batch[i])
		if generated:
			count = gen_dir_counts[category]
			gen_dir_counts[category] += 1
		else:
			count = orig_dir_counts[category]
			orig_dir_counts[category] += 1

		path = data_dir + '/{0:03d}'.format(category) + '/{0:04d}.png'.format(count)
		torchvision.utils.save_image(image, path)
	return


def train_model(denoiser, optimizer, epochs):
	denoiser.train()
	count = 0
	total_loss = 0
	start = time.time()
	train_array, test_array = [], []
	classifier.eval()

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*20)
		total_loss = 0
		count = 0
		for i, pair in enumerate(trainloader):
			if len(pair[1]) < batch_size:
				break
			train_x, train_y = pair[0], pair[1]
			trainx = train_x.to(device)
			with torch.no_grad():
				target_output = classifier(trainx)
				print (target_output.shape)

			generated_input = generate_singleinput(classifier, trainx, target_output, 0, 0)
			save = True
			if save:
				save_images(generated_input, train_y)
				save_images(trainx, train_y, generated=False)
			else:
				input = generated_input.reshape(batch_size, 3, 32, 32).permute(0, 2, 3, 1).cpu().detach().numpy()
				tx = trainx.reshape(batch_size, 3, 32, 32).cpu().permute(0, 2, 3, 1).cpu().detach().numpy()
				show_batch(input[:64, :, :, :], e)
				show_batch(tx[:64, :, :, :], e-1)
			print (f"Completed in {round(time.time() - start, 2)} seconds")
			start = time.time()

	return


def generate_singleinput(model, input_tensors, output_tensors, index, count):
	"""
	Generates an input for a given output

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int, target class index to generate
		cout: int, time step

	kwargs: 
		random_input: bool, if True then a scaled random normal distribution is used

	returns:
		None (saves .png image)
	"""

	manualSeed = 999
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)

	class_index = index

	single_input = (torch.randn(batch_size, 3, 32, 32))/20 + 0.6 # scaled normal distribution initialization
 
	single_input = single_input.to(device)
	with torch.no_grad():
		target_output = classifier(input_tensors)

	single_input = octave(single_input, target_output, 200, [2.5, 1.5], [2.0, 0.5], 0, pad=False, crop=False)

	single_input = torchvision.transforms.Resize([32, 32])(single_input)
	return single_input

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
		input_grad = output_gradient(cropped_input, target_output) # compute input gradient
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= 0.002*(start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad # gradient descent step
		single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

	return single_input


def output_gradient(input_tensor, desired_output):
	input_tensor.requires_grad = True
	output = classifier(input_tensor)
	desired_output = desired_output.to(device)
	loss = torch.sum(torch.abs(output - desired_output))
	print (loss.item())
	loss.backward() 
	gradient = input_tensor.grad
	return gradient


def denoise_input(generator, input_tensor, desired_output):
	optimizer.zero_grad()
	desired_output = torch.flatten(desired_output, start_dim=1)
	output = generator(input_tensor)
	loss = torch.sum(torch.abs(output - desired_output))
	loss.backward()
	optimizer.step()
	return output, loss.item()

 
classifier = NewResNet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True), 100).to(device)
data_dir = 'resnet_CIFAR100.pth'
classifier.load_state_dict(torch.load(data_dir))
generator = FCautoencoder(2000).to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5)
epochs = 1
train_model(generator, optimizer, epochs)
# generator_dir = 'fcnet_autoencoder_CIFAR.pth'
# torch.save(generator.state_dict(), generator_dir)
