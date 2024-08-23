import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import random
import numpy as np
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import math, random, time
import prettytable
from prettytable import PrettyTable
from datasets import load_dataset

import einops
from functools import partial 
from einops import rearrange, reduce
from safetensors.torch import load_model, save_model, load_file
from mixer_autoencoder import AutoencodingMixer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def FeedForward(dim, expansion_factor=4):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Linear(inner_dim, dim)
    )

def ConvForward(dim, expansion_factor=1):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        nn.Conv1d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Conv1d(inner_dim, dim, 1)
        )


class MixerBlock(nn.Module):

    def __init__(self, dim, length, mixer_mask=True, expand_conv=False):
        super().__init__()
        self.patch_layernorm = nn.LayerNorm(dim)
        self.seq_layernorm = nn.LayerNorm(dim)
        self.dim = dim
        self.length = length
        self.patch_ff = FeedForward(dim)
        if expand_conv:
            self.conv = ConvForward(length)
        else:
            self.conv = nn.Conv1d(length, length, 1)
        self.mixer_mask = mixer_mask
        self.expand_conv = expand_conv

    def forward(self, x: torch.tensor):
        if x.dim() > 3:
            x = rearrange(x, 'b p t f -> (b p) t f')

        # for CLM training, apply lower triangular mask to convolution weights
        if self.mixer_mask:
            if self.expand_conv:
                rearranged_shape = rearrange(self.conv[0].weight, 'f d p -> f (d p)').shape
                mask = torch.tril(torch.ones(rearranged_shape)).to(device)
                applied_mask = rearrange(self.conv[0].weight, 'f d p -> f (d p)') * mask
                self.conv[0].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

                rearranged_shape = rearrange(self.conv[2].weight, 'f d p -> f (d p)').shape
                mask = torch.tril(torch.ones(rearranged_shape)).to(device)
                applied_mask = rearrange(self.conv[2].weight, 'f d p -> f (d p)') * mask
                self.conv[2].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

            else:
                rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
                mask = torch.tril(torch.ones(rearranged_shape)).to(device)
                applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
                self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)


        residual = x
        x = self.seq_layernorm(x)
        x = self.conv(x) + residual
        residual = x
        x = self.patch_layernorm(x)
        x = self.patch_ff(x) + residual
        return x


class LanguageMixer(nn.Module):

    def __init__(self, n_vocab, dim, depth, tie_weights=False):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, dim)
        self.mixerblocks = nn.ModuleList(
            [MixerBlock(
                dim = dim,
                length = tokenized_length,
                )
            for i in range(depth)]
            ).to(device)
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        if tie_weights:
             self.wte.weight = self.lm_head.weight
        self.cel = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = input_ids
        x = x.to(device)
        x = self.wte(x)
        for block in self.mixerblocks:
            x = block(x)
        output = self.lm_head(x)
        labels = rearrange(labels, 'b p t -> b (p t)')
        output = rearrange(output, 'b t e -> b e t')
        shift_logits = output[..., :-1].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.cel(shift_logits, shift_labels)
        return loss, output

def octave(single_input, target_output, iterations, learning_rates, index):
    start_lr, end_lr = learning_rates
    original_input = single_input.clone()
    losses, i_arr = [], []

    for i in range(iterations):
        input_grad, loss = layer_gradient(model, single_input, target_output, index)
        single_input = single_input.detach()
        single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad
    return single_input

def generate_singleinput(model, target, index, lr=0.002): # 0.002 for 2048, 0.02 for the rest
    random_input = torch.randn(embedding.shape).to(device)
    single_input = octave(random_input, target, 500, [lr, lr*10], index) # lr/1 for 2048
    return single_input

def layer_gradient(model, input_tensor, target, index, cosine_metric=False):
    input_tensor.requires_grad = True
    output = a_model(input_tensor)

    if cosine_metric:
        last = 2201
        output, target = output[:, :, :last].flatten(), target[:, :, :last].flatten()
        loss = 1 - torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))
  
    else:
        loss = torch.sum(torch.abs(target[:, -1, :] - output[:, -1, :]))
        
    print (loss.item())
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()


class AbbreviatedMixer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):

        for i in range(8):
            x = self.model.encoderblocks[i](x)

        return x

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

tokenized_length = 512
dim = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoencodingMixer(n_vocab, dim, 8).float().to(device)

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

# prompt = train_text[0]['text']
# print (prompt)
prompt = 'Mario, the Idea, versus Mario, the Man.'
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer.encode(
      prompt,
      add_special_tokens=False,
      return_tensors='pt',
      truncation=True,
      padding='max_length', 
      max_length=tokenized_length,
      ).to(device)

og_model = model

# for safetensors
load_model(model, '/home/bbadger/Desktop/autoencoder_mixer_1024_n16_b32.safetensors')

# for pickled bins
# model.load_state_dict(torch.load('/home/bbadger/Desktop/tinystories_mixer_256_e1_n8/checkpoint-80000/pytorch_model.bin'))
model = AbbreviatedMixer(model)

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

count_parameters(model)

# llama_config_kwargs = {
#         'hidden_size': dim,
#         'intermediate_size': 4*dim,
#         'num_hidden_layers': 8,
#         'num_heads': 64
#     }                                       

# Initializing a LLaMA llama-70b style configuration
# configuration = LlamaConfig(**llama_config_kwargs)
                                                       
# Initializing a model from the llama-7b style configuration
# tmodel = LlamaForCausalLM(configuration).to(device)

embedding = og_model.wte(tokens)

shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
print (f'Shifted embedding distance: {torch.sum(torch.abs(embedding - shifted_embedding))}')
embedding_weight = og_model.wte.weight.float() # convert to float in case model is in 16-bit precision
inverse_embedding = torch.linalg.pinv(embedding_weight.cpu()).to(device)
print ('inverse embedding computed')
logits = torch.matmul(shifted_embedding.float(), inverse_embedding.float()) # invert embedding transformations
tokens = torch.argmax(logits, dim=2)[0]
output = tokenizer.decode(tokens)

a_model = model
a_model.eval()
with torch.no_grad():
    shifted_target_tensor = a_model(shifted_embedding).to(device)
    target_tensor = a_model(embedding).to(device)
print (f'Shifted output distance: {torch.sum(torch.abs(shifted_target_tensor - target_tensor))}')

embedding = embedding.detach()
generated_input = generate_singleinput(a_model, target_tensor, 0)
g_input = generated_input

generated_target_tensor = a_model(g_input).to(device)
print (f'Generated output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')
logits = torch.matmul(generated_input, inverse_embedding)

tokens = torch.topk(logits, 5)[1][0] # indicies of topk of tensor [length, topk_tokens]

for i in range(5):
    output = tokenizer.decode([o[i] for o in tokens])
    print (output)
    break

print ('\n')