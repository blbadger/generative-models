import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file



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

	def __init__(self, dim, length, mixer_mask=True, expansion_factor=4, expand_conv=True):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim, expansion_factor=expansion_factor)
		self.expand_conv = expand_conv
		if self.expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1)
		
		# for CLM training, apply lower triangular mask to convolution weights
		self.mixer_mask = mixer_mask

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		if self.mixer_mask:
			if self.expand_conv:
				masked_conv0 = nn.Parameter(rearrange(torch.tril(rearrange(self.conv[0].weight, 'f d p -> f (d p)')), 'f (d p) -> f d p', p=1))
				masked_conv2 = nn.Parameter(rearrange(torch.tril(rearrange(self.conv[2].weight, 'f d p -> f (d p)')), 'f (d p) -> f d p', p=1))
				self.conv[0].weight = masked_conv0
				self.conv[2].weight = masked_conv2
			else:
				self.conv.weight = torch.nn.Parameter(rearrange(self.conv.weight, 'f d p -> f (d p)'))
				self.conv.weight = torch.nn.Parameter(torch.tril(self.conv.weight))
				self.conv.weight = torch.nn.Parameter(rearrange(self.conv.weight, 'f (d p) -> f d p', p=1))
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
			self.lm_head.weight = self.wte.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = self.lm_head(x)
		output = rearrange(output, 'b t e -> b e t')
		return [], output

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
# model = AutoModel.from_pretrained('/home/bbadger/Desktop/tinystories_mixer/checkpoint-15000')
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

# barebones MLP mixer, expects an embedding on input tokens
tokenized_length = 512
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 12).float().to(device)
load_model(model, '/home/bbadger/Desktop/tinystories_mixer_masked/checkpoint-194000/model.safetensors')

prompt = prompt = '''Once upon a time there was a tree named Barky. He '''

tokens = tokenizer.encode(
				prompt,
				add_special_tokens=False,
				return_tensors='pt',
				padding='max_length',
				max_length=512
			)
# print (tokens)

fout = []
for i in range(50):
	output = model(tokens)[1]
	last_output = output[:, :, -1]
	output_index = torch.topk(last_output, dim=-1, k=1).indices
	fout.append(int(output_index))
	output_token = output_index.to('cpu')
	tokens = torch.cat((tokens[:, 1:], output_token), dim=-1)

print (tokenizer.decode(fout))