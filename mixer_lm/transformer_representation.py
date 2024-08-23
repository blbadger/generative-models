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

import einops
from functools import partial 
from einops import rearrange, reduce
from safetensors.torch import load_model, save_model, load_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def octave(single_input, target_output, iterations, learning_rates, index):
    start_lr, end_lr = learning_rates
    original_input = single_input.clone()
    losses, i_arr = [], []

    for i in range(iterations):
        input_grad, loss = layer_gradient(model, single_input, target_output, index)
        single_input = single_input.detach()
        single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad
    return single_input

def generate_singleinput(model, target, index, lr=0.02): # 0.01
    random_input = torch.randn(embedding.shape).to(device)
    single_input = octave(random_input, target, 1000, [lr, lr/10], index)
    return single_input

def layer_gradient(model, input_tensor, target, index, cosine_metric=False):
    input_tensor.requires_grad = True
    output = a_model(input_tensor)

    if cosine_metric:
        last = 2201
        output, target = output[:, :, :].flatten(), target[:, :, :].flatten()
        loss = 1 - torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))
  
    else:
        loss = torch.sum(torch.abs(target[:, :, :] - output[:, :, :]))
        
    print (loss.item())
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()


class AbbreviatedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Matrix mult instead of embedding to prevent type incompatibility
        position_ids = torch.tensor([[i for i in range(x.shape[1])]])

        for i in range(1):
            x = self.model.model.layers[i](x, position_ids=position_ids)[0]

        return x

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompt = 'Mario, the Idea, versus Mario, the Man'
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer.encode(
      prompt,
      add_special_tokens=False,
      return_tensors='pt',
      ).to(device)


dim = 512
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_attention_heads': 2,
    'vocab_size': 4096
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()

# for safetensors
load_model(model, '/home/bbadger/Desktop/tinystories/tinystories_llama_512_h2/checkpoint-4000/model.safetensors')
og_model = model

model = AbbreviatedModel(model).to(device)

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

embedding = og_model.model.embed_tokens(tokens)

shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
print (f'Shifted embedding distance: {torch.sum(torch.abs(embedding - shifted_embedding))}')
embedding_weight = og_model.model.embed_tokens.weight.float() # convert to float in case model is in 16-bit precision
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

print ('\n')