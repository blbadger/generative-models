import os
import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file
import json
import numpy as np
import random
from datasets import Dataset, load_from_disk, load_dataset
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

	def __init__(self, n_vocab, dim, depth, tie_weights=False, prebatched_input=True):
		super().__init__()
		self.prebatched_input = prebatched_input
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

	def forward(self, input_ids, **kwargs):
		x = input_ids
		if self.prebatched_input:
			x = x.squeeze(0) # p b t -> b t
		x = x.to(device)
		x = self.wte(x)
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		loss = infoNCEloss(x, matching_index=matching_index)
		return loss, output

def infoNCEloss(output, matching_index=None):
	"""
	Implements Noise-Contrastive Loss. Assumes that there is one positive pair per batch and all 
	the rest are negative samples.

	"""
	match_embedding = output[0, :, -1] # b t e shape
	summary_embedding = output[matching_index, :, -1]
	nonmatch_embeddings = torch.cat(output[1:matching_index, :, -1], output[matching_index+1:, :, -1], dim=0)
	codists = torch.exp(torch.cos(summary_embedding, nonmatch_embedding) / 0.01) # temperature=0.1

	nonmatching_cos = F.normalize(summary_embedding, p=2, dim=1) \
					@ F.normalize(nonmatch_embedding, p=2, dim=1).T

	nondists = torch.sum(torch.exp(nonmatching_cos), dim=0)
	loss = torch.sum(-torch.log(codists / (codists + nondists)))
	return loss


class RetrievalDataset(torch.utils.data.Dataset):

	def __init__(self, text_tokens, summary_tokens, batch_size=64, replace=False):
		self.summary_tokens = summary_tokens
		self.text_tokens = text_tokens
		self.context_length = len(self.summary_tokens[0]['input_ids'])
		self.prob_weights = torch.ones(self.context_length)
		self.allocated_input = torch.zeros((batch_size, self.context_length))
		self.replace = replace
		self.batch_size = batch_size

	def __getitem__(self, idx):
		input = torch.zeros((self.batch_size, self.context_length)) # b t shape
		input[0] = self.summary_tokens[idx]['input_ids']
		self.prob_weights[idx] = 0
		indices = torch.multinomial(self.prob_weights, self.n_context-1, replacement=self.replace)
		self.prob_weights[idx] = 1
		input[1:] = self.text_tokens[indices]['input_ids']
		target_index = random.randint(1, self.n_context-1) # random index to put target embedding
		matching_target = self.text_tokens[idx]['input_ids'] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index-1, dtype=torch.long)
		retrieval_dict = {'input_ids': input, 'matching_index': labels} # results in p b t shape upon load
		return retrieval_dict

	def __len__(self):
		return len(self.summary_tokens)
  
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_context = tokenized_length
# initialize retrieval model
retrieval_model = LanguageMixer(512, 16, n_context)

# expects left padding for both text and summary
text_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-left"
summary_path = "/home/bbadger/Desktop/contrastive-summaries-fineweb-lpad-200k"
split_index = 180000
text_tokens = load_from_disk(text_path, keep_in_memory=None)
summary_tokens = load_from_disk(summary_path, keep_in_memory=None)
train_dataset = RetrievalDataset(text_tokens, summary_tokens)
test_dataset = RetrievalDataset(text_tokens, summary_tokens)
print ('training begun')

pad_token = int(tokenizer.encode(tokenizer.pad_token)[-1])
training_arguments = transformers.TrainingArguments(
	num_train_epochs=200,
	per_device_train_batch_size=1, # actually defined in dataset subclass
	per_device_eval_batch_size=1, # actually defined in dataset subclass
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/contrastive_mixer_fineweb_512_n8_b128',
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
