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
		self.image_name_ls = images[:12]

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

batch_size = 512 # global variable
image_size = 32
channels = 3

# data_dir = pathlib.Path('../nnetworks/landscapes',  fname='Combined')
# train_data = ImageDataset(data_dir, image_type='.jpg')
# dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
# print (len(dataloader)) 

transform = transforms.Compose([
			# transforms.Resize((image_size, image_size)),
			transforms.ToTensor()
])

def npy_loader(path):
	sample = torch.from_numpy(np.load(path))
	sample = sample.permute(0, 3, 2, 1)
	#270* rotation
	for i in range(3):
		sample = torch.rot90(sample, dims=[2, 3])
	return sample / 255.

# path = pathlib.Path('../lsun_churches/churches/church_outdoor_train_lmdb_color_64.npy',  fname='Combined')

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
		self.fc = nn.Linear(512 * 4, n_output)
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=2, bias=False)

	def forward(self, x):
		x = self.model.conv1(x)
		# x = self.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		# x = self.model.layer2(x)
		# x = self.model.layer3(x)
		# x = self.model.layer4(x)

		# x = self.model.avgpool(x)
		# x = torch.flatten(x, 1)
		# x = self.fc(x)
		# x = x.reshape(batch_size, channels, image_size, image_size)
		return x

class SingleEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(128*128*3, starting)
		self.d5 = nn.Linear(starting, 128*128*3)
		self.gelu = nn.GELU()


	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		out = self.d5(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out

class SmallFCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*channels, starting)
		self.d1 = nn.Linear(starting, starting//4)
		self.d2 = nn.Linear(starting//4, starting//8)
		self.d3 = nn.Linear(starting//8, starting//4)
		self.d4 = nn.Linear(starting//4, starting)
		self.d5 = nn.Linear(starting, 32*32*channels)
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


class SmallerFCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*channels, starting)
		self.d1 = nn.Linear(starting, starting//8)
		self.d2 = nn.Linear(starting//8, starting//16)
		self.d3 = nn.Linear(starting//16, starting//8)
		self.d4 = nn.Linear(starting//8, starting)
		self.d5 = nn.Linear(starting, 32*32*channels)
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

class SmallDeepFCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*channels, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting//4)
		self.d3 = nn.Linear(starting//4, starting//8)
		self.d4 = nn.Linear(starting//8, starting//12)
		self.d5 = nn.Linear(starting//12, starting//16)
		self.d6 = nn.Linear(starting//16, starting//12)
		self.d7 = nn.Linear(starting//12, starting//8)
		self.d8 = nn.Linear(starting//8, starting//4)
		self.d9 = nn.Linear(starting//4, starting//2)
		self.d10 = nn.Linear(starting//2, starting)
		self.d11 = nn.Linear(starting, 32*32*channels)
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

		for i in range(1, 11):
			dense_layer = eval('self.d{}'.format(i))
			out = dense_layer(out)
			out = self.gelu(out)

		out = self.d11(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out


class FCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3, starting)
		self.dense_layers = [0]*5
		for i in range(5):
			self.dense_layers[i] = nn.Linear(starting, starting).to(device)
		self.d6 = nn.Linear(starting, 32*32*3)
		self.bn1 = nn.BatchNorm1d(starting)
		self.bn2 = nn.BatchNorm1d(starting)
		self.bn3 = nn.BatchNorm1d(starting)
		self.bn4 = nn.BatchNorm1d(starting)
		self.bn5   = nn.BatchNorm1d(starting)
		self.layernorm1 = nn.LayerNorm(starting)
		self.layernorm2 = nn.LayerNorm(starting)
		self.layernorm3 = nn.LayerNorm(starting)
		self.layernorm4 = nn.LayerNorm(starting)
		self.gelu = nn.GELU()

	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		
		out = self.input_transform(input_tensor)
		out = self.bn1(self.gelu(out))

		for i in range(5):
			out = self.dense_layers[i](out)
			out = self.gelu(out)

		out = self.d6(out)
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
	plt.savefig('image{0:04d}.png'.format(count), dpi=300, transparent=True)
	print ('Image Saved')
	plt.close()
	return 

# model = FCEncoder(5000, 3).to(device) 
# model = SmallDeepFCEncoder(4000, 3).to(device)
# model = SmallFCEncoder(4000, 3).to(device)
model = FCEncoder(4000, 3).to(device)
# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=3, init_features=32, pretrained=False).to(device)

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

# model = unet_noresiduals.UNet(n_channels=3, n_classes=3).to(device)
# resnet = NewResnet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True), 32*32*3).to(device)
# model = SingleEncoder(30000, channels=3).to(device)


optimizer = Adam(model.parameters(), lr=1e-4) 
loss_fn = torch.nn.MSELoss()

def train_autoencoder():
	epochs = 1000
	for epoch in range(epochs):
		start_time = time.time()
		total_loss = 0
		for step, batch in enumerate(dataloader):
			# if step > 20:
			# 	break
			if len(batch[0]) < batch_size:
				break 
			# alpha = random.random() 
			# batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
			optimizer.zero_grad()
			batch = batch[0].to(device) # discard class labels
			output = model(batch)
			loss = loss_fn(output, batch) # + loss_fn(gen_res, real_res)
			total_loss += loss.item()
	 
			loss.backward()
			optimizer.step()

		print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
		print (f"Average Loss: {round(total_loss / step, 5)}")
		torch.save(model.state_dict(), 'fcnet_autoencoder_bn_cifar.pth')

		if epoch % 10 == 0:
			batch = next(iter(dataloader))[0].to(device)
			gen_images = model(batch).cpu().permute(0, 2, 3, 1).detach().numpy()
			show_batch(gen_images, count=epoch, grayscale=False, normalize=False)


def interpolate_latent():
	data = iter(dataloader)
	batch1 = next(data).to(device)
	batch2 = next(data).to(device)
	random = torch.normal(0.5, 0.2, batch1.shape).to(device) 
	for i in range(61):
		alpha = 1 - i / 30
		if i <= 30:
			beta = i / 30
		else:
			beta = abs(2 - i / 30)
		batch = alpha * batch1 + (1 - alpha)* batch2 + 2 * beta * random 
		gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i, grayscale=False, normalize=True)

	return

# interpolate_latent() 

class UnetDecoder(nn.Module):
    def __init__(self, unet, dummy_input):
        super().__init__()
        self.unet = unet
        self.dummy_input = dummy_input

    def forward(self, x):
        x1 = self.unet.inc(self.dummy_input)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)
        x = self.unet.up1(x, x4) # changed to take hidden input
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        logits = self.unet.outc(x)
        return logits

class Hidden(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module.hidden

    def forward(self, x):
        hidden = x
        x = self.module.hidden_out1(x)
        x = self.module.hidden_out2(x)
        x = x.reshape((len(x), 1024, 4, 4))
        x = self.module.conv_transpose(x)
        return x

class UnetHiddenDecoder(nn.Module):
    def __init__(self, unet, dummy_input):
        super().__init__()
        self.unet = unet
        self.dummy_input = dummy_input
        self.hidden = Hidden(unet)

    def forward(self, x):
        x1 = self.unet.inc(self.dummy_input)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)

        x = self.hidden(x)
        x = self.unet.up1(x, x4) # changed to take hidden input
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        logits = self.unet.outc(x)
        return logits


@torch.no_grad()
def random_manifold_walk():
	data = iter(dataloader)
	batch = next(data).to(device)[0]
	output, hidden = model(batch)
	gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=0, grayscale=False, normalize=True)

	unet_decoder = UnetDecoder(model, batch)
	og_hidden = hidden
	for i in range(300):
		random = torch.normal(0, 0.2, hidden.shape).to(device)
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
	batch = next(iter(dataloader))[0]
	original_batch = batch
	show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=False, normalize=False)
	# alpha = 0.5
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

	alpha = 0.3
	batch = original_batch

	original = batch[0]
	original_output = model(batch.to(device))
	batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
	transformed = batch
	transformed_output = model(batch.to(device))

	shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(shown, count=100, grayscale=False, normalize=True)
	gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=99, grayscale=False, normalize=True)
	input_distance = torch.sum((original - transformed)**2)**0.5
	output_distance = torch.sum((original_output - transformed_output)**2)**0.5
	print (f'L2 Distance on the Input after Gaussian Noise: {input_distance}')
	print (f'L2 Distance on the Autoencoder Output after Gaussian Noise: {output_distance}')


@torch.no_grad()
def generate_with_noise():
	batch = next(iter(dataloader))[0]
	alpha = 0
	batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
	for i in range(80):
		alpha = i / 80
		gen_images = model(batch.to(device))
		show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)
		batch = alpha * gen_images + (1-alpha) * torch.normal(0.6, 0.2, batch.shape).to(device) 

	return batch


def find_analogues(input):
	batch = next(iter(dataloader))
	images = []
	for step, batch in enumerate(dataloader):
		if step > 20:
			break
		batch = batch[0]
		for i in range(512):
			images.append(batch[i, :, :, :])

	min_distance = np.inf
	for image in images:
		if torch.sum((input - image.to(device))**2)**0.5 < min_distance:
			closest_image = image
			min_distance = torch.sum((input - image[0].to(device))**2)**0.5 

	closest_image = closest_image.cpu().permute(1, 2, 0).detach().numpy()
	# plt.figure(figsize=(15, 15))
	# plt.imshow(closest_image)
	# plt.tight_layout()
	# plt.savefig('closest_image.png', dpi=300, transparent=True)

	input_batch = input.cpu().permute(1, 2, 0).detach().numpy(), closest_image
	length, width = 1, 2
	for n in range(length*width):
		ax = plt.subplot(length, width, n+1)
		plt.axis('off')
		plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.savefig('closest_pair.png', dpi=300, transparent=True)
	print ('Image Saved')
	plt.close()
	return 
	plt.close()
	return

if __name__ == '__main__':
	count_parameters(model)
	# batch = next(iter(data loader)).cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(batch, count=999, grayscale=False, normalize=False)
	# model.load_state_dict(torch.load('fcnet_smallautoencoder_cifar.pth')) 
	train_autoencoder()

	# model.eval()
	# random_manifold_walk()
	observe_denoising()
	batch = generate_with_noise()
	# data = iter(dataloader)
	# batch = next(data)[0].to(device)
	# show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy())
	find_analogues(batch[0])

