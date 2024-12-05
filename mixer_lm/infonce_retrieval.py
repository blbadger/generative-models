import os
import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file
import json
import numpy as np
import random
from datasets import Dataset
from safetensors.torch import safe_open
from tqdm import tqdm

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

	def __init__(self, dim, length=512, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1, padding='same')
		self.expand_conv = expand_conv
		#heads = 4
		#self.mixerhead = MixerHead(1024, 512, 512, heads)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
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
			masked_conv = torch.tril(rearrange(self.conv.weight, 'f d p -> p f d'))
			self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

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
				expand_conv=False
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			 self.wte.weight = self.lm_head.weight
		self.infoNCEloss = infoNCEloss()

	def forward(self, input_ids, **kwargs):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.infoNCE(shift_logits, shift_labels)
		return loss, output

def infoNCEloss(output, matching_index):
	"""
	Implements Noise-Contrastive Loss. Assumes that there is one positive pair per batch and all 
	the rest are negative samples.

	"""
	codists = torch.exp(torch.cos(output[matching_index, ...], target[0], dim=0)/0.01) # temperature=0.1
	nondists = torch.sum(torch.exp(torch.cos(output, target[1:])))
	loss = torch.sum(- torch.log(codists / (codists + nondists)))
	return loss

def input_batch(input_tokens):
	embeddings = []
	pad_token = int(tokenizer.encode(tokenizer.pad_token)[-1])
	for i in range(0, len(input_tokens)):
		if i % 1000 == 0:
			print (i)
		output = gen_model(
			torch.tensor(input_tokens[i]).unsqueeze(0).to(device),
			output_hidden_states=True
		)
		t = 0
		while (t in range(len(input_tokens[i])-1) and int(input_tokens[i][t]) != pad_token):
			t += 1
		t -= 1
		last_hidden_layers = output.hidden_states[-1][..., t, :].detach().to('cpu')
		# expects the model's output to be the last hidden layer
		embeddings.append(last_hidden_layers)

	embeddings = torch.stack(embeddings).squeeze(1)
	return embeddings

@torch.no_grad()
def embed_input(input_tokens, pad_token):
	embeddings = [] 
	
	last_hidden_layers = gen_model(
		torch.tensor(input_tokens[i])
	)[..., t, :].detach().to('cpu')
	# expects the model's output to be the last hidden layer
	embeddings.append(last_hidden_layers)
	embeddings = torch.stack(embeddings).squeeze(1)
	return embeddings


pad_token = int(tokenizer.encode(tokenizer.pad_token)[-1])







tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'



n_context = 512
train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings, n_context=n_context, replace=True, pre_index=False)
test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context, replace=True)
print (len(target_test_embeddings), len(query_test_embeddings))

# initialize retrieval model
retrieval_model = RetrievalMixer(512, 8, n_context)
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=200,
	per_device_train_batch_size=128,
	per_device_eval_batch_size=128,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/fineweb_retrieval_mixer_512_n8_200k_c32',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True
)

trainer = transformers.Trainer(
	model=retrieval_model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments
)

retrieval_model.train()
trainer.train()
