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
scaler = torch.amp.GradScaler("cuda" , enabled=True)
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
        #image = torchvision.transforms.CenterCrop([728, 728])(image)
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

def load_model(model):
    model = model.to('cpu')
    checkpoint = torch.load('/home/bbadger/Desktop/churches_unetdeepwide/epoch_490', map_location=torch.device('cpu'))
    reformatted_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        reformatted_checkpoint[key[7:]] = value
    model.load_state_dict(reformatted_checkpoint)
    del checkpoint
    return model.to("cpu")


batch_size = 32
image_size = 128
channels = 3

# model = UNetWide(3, 3).to(device)
model = UNetDeepWide(3, 3)
model_dtype = torch.float16
# model = UNetWideHidden(3, 3).to(device)
loss_fn = torch.nn.MSELoss(reduction='none')
# loss_fn = torch.nn.BCELoss()
cosine_loss = torch.nn.CosineSimilarity(dim=0)
count_parameters(model)
print ('model_loaded')

def train_autoencoder(model, dataset='churches', epochs=5000):
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
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            for step, batch in enumerate(dataloader):
                if len(batch) < batch_size:
                    break 
                
                batch = batch.to(device_id) # discard class labels
                output = ddp_model(batch) 
                loss = loss_fn(output, batch)

            loss_size = loss.shape,
            loss = torch.mean(loss)
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()

        if rank == 0:
            checkpoint_path = f'/home/bbadger/Desktop/landscapes_unetwidedeep/epoch_{epoch}'
            if epoch % 1000 == 0: torch.save(ddp_model.state_dict(), checkpoint_path)
            tqdm.write(f"Epoch {epoch} completed in {time.time() - start_time} seconds")
            tqdm.write(f"Average Loss: {round(total_loss / step, 5)}")
            tqdm.write(f"Loss shape: {loss_size}")
        dist.barrier()

    dist.destroy_process_group()
 
if __name__ == '__main__':
    train_autoencoder(model, dataset='landscapes')