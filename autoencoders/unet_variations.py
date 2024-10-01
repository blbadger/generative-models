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
import torch.multiprocessing as mp
from torch.optim import Adam
import torchvision
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from unet_noresiduals import UNet_hidden, UNetWide, UNetDeep, UNetDeepWide, UNetWideHidden, UNet
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from prettytable import PrettyTable

device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, device_idx, transform=None, target_transform=None, image_type='.png'):
        # specify image labels by folder name 
        self.img_labels = [item.name for item in img_dir.glob('*')]

        # split images among devices: assumes one label per image
        gpu_count = torch.cuda.device_count()
        start = (len(self.img_labels) // gpu_count) * device_idx
        end = start + (len(self.img_labels) // gpu_count)

        # construct image name list: randomly sample images for each epoch
        images = list(img_dir.glob('*' + image_type))[start:end]
        self.image_name_ls = images
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints, torchvision.io.ImageReadMode.GRAY
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

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    sample = sample.permute(0, 3, 2, 1)
    #270* rotation
    for i in range(3):
        sample = torch.rot90(sample, dims=[2, 3])
    return sample / 255.

def show_batch(input_batch, count=0, grayscale=False, normalize=True, tag=None):
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
    plt.savefig(f'{tag}_{count:04d}.png', dpi=300, transparent=True)
    print ('Image Saved')
    plt.close()
    return
  
batch_size = 32
image_size = 64
channels = 3

# model = UNet(3, 3)
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# model = UNetWide(3, 3).to(device)
model = UNetDeepWide(3, 3)
# model = UNetWideHidden(3, 3).to(device)
loss_fn = torch.nn.MSELoss(reduction='none')
# loss_fn = torch.nn.BCELoss()
cosine_loss = torch.nn.CosineSimilarity(dim=0)
count_parameters(model)

def load_model(model):
    model = model.to('cpu')
    checkpoint = torch.load('/home/bbadger/Desktop/churches_unetdeepwide/24')
    reformatted_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        reformatted_checkpoint[key[7:]] = value
    model.load_state_dict(reformatted_checkpoint)
    del checkpoint
    return model.to("cpu")

#model = load_model(model)

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

# batch = next(iter(dataloader)).cpu().permute(0, 2, 3, 1).detach().numpy()
# model.load_state_dict(torch.load('/home/bbadger/Downloads/499'))
checkpoint = torch.load('/home/bbadger/Desktop/499')
reformatted_checkpoint = OrderedDict()
for key, value in checkpoint.items():
    reformatted_checkpoint[key[7:]] = value
# del checkpoint
model.load_state_dict(reformatted_checkpoint)
# train_autoencoder(model, dataset='curches')
model = model.to(device)

# model.eval() switches batch norm from online statistics to saved: do not use for batchnormed-trained autoencoders
# model.eval()


path = pathlib.Path('/home/bbadger/Downloads/church_outdoor_train_lmdb_color_64.npy', fname='Combined')
dataset = npy_loader(path)
dset = torch.utils.data.TensorDataset(dataset)
dataloader = torch.utils.data.DataLoader(dataset[:500], batch_size=batch_size, shuffle=True)


@torch.no_grad()
def observe_denoising(alpha):
    batch = next(iter(dataloader))
    original_batch = batch
    show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=0, grayscale=False, normalize=False, tag='no_noise')
    # alpha = 0.5
    # batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)

    # original = batch[0]
    # original_output = model(batch.to(device))[0][0]
    # batch = torchvision.transforms.GaussianBlur(19, 8)(batch)
    # transformed = batch[0]
    # transformed_output = model(batch.to(device))[0][0]

    # shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
    # show_batch(shown, count=1000, grayscale=False, normalize=False)
    # gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
    # show_batch(gen_images, count=999, grayscale=False, normalize=False)
    # input_distance = torch.sum((original - transformed)**2)**0.5
    # output_distance = torch.sum((original_output - transformed_output)**2)**0.5
    # print (f'L2 Distance on the Input after Blurring: {input_distance}')
    # print (f'L2 Distance on the Autoencoder Output after Blurring: {output_distance}')

    batch = original_batch

    original = batch[0]
    original_output = model(batch.to(device))[0][0]
    batch = alpha * batch + (1-alpha) * torch.normal(0.5, 0.2, batch.shape)
    transformed = batch[0]
    transformed_output = model(batch.to(device))[0][0]

    shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(shown, count=0, grayscale=False, normalize=False, tag='noised')
    gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
    show_batch(gen_images, count=1, grayscale=False, normalize=False, tag='noise_removed')
    input_distance = torch.sum((original - transformed)**2)**0.5
    output_distance = torch.sum((original_output - transformed_output)**2)**0.5
    print (f'L2 Distance on the Input after Gaussian Noise: {input_distance}')
    print (f'L2 Distance on the Autoencoder Output after Gaussian Noise: {output_distance}')

alpha = 0.9
observe_denoising(alpha)

@torch.no_grad()
def generate_with_noise():
    batch = next(iter(dataloader))
    alpha = 0
    batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
    for i in range(30):
        alpha = i / 20
        gen_images = model(batch.to(device))
        batch = alpha * gen_images + (1-alpha) * torch.normal(0.7, 0.2, batch.shape).to(device) 
        show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=True, tag='denoising_gen')

generate_with_noise()

@torch.no_grad()
def generate_with_increasing_resolution(starting_resolution=128):
    batch = next(iter(dataloader))
    alpha = 0
    batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
    for i in range(30):
        if i == 20:
            batch = torchvision.transforms.Resize(256)(batch)
        # if i == 40:
        #     batch = torchvision.transforms.Resize(512)(batch)
        alpha = i / 40
        gen_images = model(batch.to(device))
        batch = alpha * gen_images + (1-alpha) * torch.normal(0.7, 0.2, batch.shape).to(device) 
        show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)

# generate_with_increasing_resolution()


# new_vision = model
# alpha = 0.1
# batch = next(iter(dataloader))
# image = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
# # image = image[0].reshape(1, 3, 128, 128).to(device)
# target_tensor = new_vision(image)

# target_tensor = target_tensor.detach().to(device)
# plt.figure(figsize=(10, 10))
# image_width = len(image[0][0])
# target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
# plt.imshow(target_input)
# plt.axis('off')
# plt.savefig('target_image', bbox_inches='tight', pad_inches=0.1)
# plt.close()

# modification = torch.randn(1, 3, 128, 128)/18
# modification = modification.to(device)
# modified_input = image + modification
# modified_output = new_vision(modified_input)
# print (f'L2 distance between original and shifted inputs: {torch.sqrt(torch.sum((image - modified_input)**2))}')
# print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')

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
# image = image[0].reshape(1, 3, image_size, image_size).to(device)
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
    if random_input:
        single_input = (torch.randn(1, 3, image_size, image_size))/20 + 0.7 # scaled normal distribution initialization
    else:
        single_input = input_tensors

    iterations = 1000 
    single_input = single_input.to(device)
    single_input = single_input.reshape(1, 3, image_size, image_size)
    original_input = torch.clone(single_input).reshape(3, image_size, image_size).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)

    single_input = octave(single_input, target_output, iterations, [0.1, 0.1], [2.4, 0.4], 0, pad=False, crop=False)

    output = model(single_input).to(device)
    print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')
    target_input = torch.tensor(target_input).reshape(1, 3, image_size, image_size).to(device)
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
    loss = 0.01*torch.sum(torch.abs(target_tensor - output)) # target_tensor is the desired activation
    loss.backward()
    gradient = input_tensor.grad
    return gradient


# generate_singleinput(new_vision, image, [], 0, 0, image, random_input=True)
