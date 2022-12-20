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

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
	"""
	Creates a dataset from images classified by folder name.  Random
	sampling of images to prevent overfitransformsing
	"""

	def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
		# specify image labels by folder name 
		self.img_labels = [item.name for item in data_dir.glob('*')]

		# construct image name list: randomly sample 400 images for each epoch
		images = list(img_dir.glob('*/*' + image_type))
		random.shuffle(images)
		self.image_name_ls = images[:800]

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
		image = torchvision.transforms.Resize(size=[28, 28])(image) 

		# assign label to be a tensor based on the parent folder name
		label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

		# convert image label to tensor
		label_tens = torch.tensor(self.img_labels.index(label))
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label_tens



transform = transforms.Compose(
	[transforms.ToTensor()])

stats = ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
train_transforms= transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.ToTensor()])

batch_size = 256
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

class FCnet(nn.Module):

	def __init__(self, starting_size):

		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting//4)
		self.d3 = nn.Linear(starting//4, starting//8)
		self.d4 = nn.Linear(starting//8, 100)
		self.gelu = nn.GELU()
		self.softmax = nn.Softmax()

	def forward(self, input_tensor):
		input_tensor = torch.flatransformsen(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		out = self.d1(out)
		out = self.gelu(out)

		out = self.d2(out)
		out = self.gelu(out)

		out = self.d3(out)
		out = self.gelu(out)

		out = self.d4(out)
		return out



class FCnetrev(nn.Module):

	def __init__(self, starting_size):

		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3, starting)
		self.d1 = nn.Linear(int(starting), int(starting))
		self.d2 = nn.Linear(int(starting), int(starting))
		self.d3 = nn.Linear(int(starting), int(starting))
		self.d4 = nn.Linear(int(starting), 10)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()

	def forward(self, input_tensor):
		input_tensor = torch.flatransformsen(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.relu(out)

		out = self.d1(out)
		out = self.relu(out)

		out = self.d2(out)
		out = self.relu(out)

		out = self.d3(out)
		out = self.relu(out)

		out = self.d4(out)
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
	for n in range(16*16):
		ax = plt.subplot(16, 16, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray_r')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.show()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

def train_model(dataloader, model, optmizer, loss_fn, epochs):
	model.train()
	count = 0
	total_loss = 0
	start = time.time()
	train_array, test_array = [], []

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*20)
		total_loss = 0
		count = 0
		for pair in trainloader:
			train_x, train_y = pair[0], pair[1]
			trainx = train_x.to(device)
			output = model(trainx)
			loss = loss_fn(output.to(device), train_y.to(device))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss
			count += 1

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Average Loss: {ave_loss:.04}")
		test_model(testloader, model)
		start = time.time()

	return

def test_model(dataloader, model):
	model.eval()
	correct, count = 0, 0
	batches = 0
	for batch, (x, y) in enumerate(dataloader):
		x = x.to(device)
		predictions = model(x)
		_, predicted = torch.max(predictions.data, 1)
		count += len(y)
		correct += (predicted == y.to(device)).sum().item()
		batches += 1

	print (f'Test Accuracy: {correct / count}')
	return correct / count


class NewResNet(nn.Module):

	def __init__(self, model, num_classes):
		super().__init__()
		self.model = model
		self.inplanes = 64
		self.fc = nn.Linear(512 * 4, num_classes)
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=2, bias=False)

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
		return x

train = True
if train:
	train_accuracies, test_accuracies = [], []
	epochs = 20 
	loss_fn = nn.CrossEntropyLoss()
	model = NewResNet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True), 100)
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print (total_params)
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	train_model(trainloader, model, optimizer, loss_fn, epochs)
	trainloader = trainloader
	testloader = testloader

	data_dir = 'resnet_CIFAR100.pth'
	torch.save(model.state_dict(), data_dir)

model = NewResNet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True), 100)
data_dir = 'resnet_CIFAR100.pth'
model.load_state_dict(torch.load(data_dir))

def save_image(single_input, count, output):
    """
    Saves a .png image of the single_input tensor

    Args:
        single_input: torch.tensor of the input 
        count: int, class number

    Returns:
        None (writes .png to storage)
    """

    print (count)
    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    predicted = int(torch.argmax(output))
    print (predicted)
    target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(target_input)
    images_dir = './CIFAR100_generated'
    plt.savefig("{}".format(images_dir) + "/Class {0:04d}".format(count), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return

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


def octave(single_input, target_output, iterations, learning_rates, sigmas, size, pad=False, crop=True, index=0):
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

    for i in range(iterations):
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])
        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(model, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations) * input_grad # gradient descent step
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))
        if pad:
            single_input = torchvision.transforms.Pad([1, 1], fill=0.7)(single_input)

    return single_input



def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
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

    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    class_index = index

    if random_input:
        single_input = (torch.randn(1, 3, 32, 32))/20 + 0.7 # scaled normal distribution initialization

    else:
        single_input = input_tensors[0]
 
    single_input = single_input.to(device)
    original_input = torch.clone(single_input).reshape(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy()
    single_input = single_input.reshape(1, 3, 32, 32)
    original_input = torch.clone(single_input).reshape(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)

    pad = False
    if pad:
        single_input = octave(single_input, target_output, 220, [6, 5], [2.4, 0.8], 0, pad=True, crop=False)
    else:
        single_input = octave(single_input, target_output, 220, [2.5, 1.5], [2.4, 0.8], 0, pad=False, crop=False)

    single_input = torchvision.transforms.Resize([50, 50])(single_input)
    single_input = octave(single_input, target_output, 200, [1.5, 0.3], [1.5, 0.4], 40, pad=False, crop=True) 

    single_input = torchvision.transforms.Resize([60, 60])(single_input)
    single_input = octave(single_input, target_output, 100, [1.5, 0.4], [1.5, 0.4], 45, pad=False, crop=True)

    output = model(single_input)
    save_image(single_input, index, output)
    return single_input


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
    output = model(input_tensor).to(device)
 
    loss = 0.5 * (200 - output[0][int(desired_output)])
    loss.backward()
    gradient = input_tensor.grad

    return gradient


def generate_inputs(model, count=0):
    """
    Generate the output of each desired class

    Args:
        model: torch.nn.Module object of interest

    kwargs:
        count: int, time step 

    Returns:
        None (saves .png images to storage)
    """

    for i in range(100):
        generate_singleinput(model, [], [], i, count, random_input=True)

    return


def show_batch(input_batch, count=0, grayscale=False):
    """
    Show a batch of images with gradientxinputs superimposed

    Args:
        input_batch: arr[torch.Tensor] of input images
        output_batch: arr[torch.Tensor] of classification labels
        gradxinput_batch: arr[torch.Tensor] of attributions per input image
    kwargs:
        individuals: Bool, if True then plots 1x3 image figs for each batch element
        count: int

    returns:
        None (saves .png img)

    """

    plt.figure(figsize=(15, 15))
    for n in range(16):
        ax = plt.subplot(4, 4, n+1)
        plt.axis('off')
        if grayscale:
            plt.imshow(input_batch[n], cmap='gray')
        else:
            plt.imshow(input_batch[n])
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig('transformed_flowers{0:04d}.png'.format(count), dpi=410)
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


model.eval()
model.to(device)
generate_inputs(model, 0)