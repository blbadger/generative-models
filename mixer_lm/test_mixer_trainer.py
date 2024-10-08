import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import prettytable
from prettytable import PrettyTable

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
from transformers import LlamaConfig, LlamaForCausalLM


def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

def ConvForward(dim, expansion_factor=2):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1)
		)


class MixerBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=True, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.ref_conv = nn.Conv1d(length, 1, 1)
			self.conv = nn.Conv1d(length, length, 1)
			# with torch.no_grad():
			# 	for i in range(length):
			# 		self.conv.weight[i, :, :] = self.ref_conv.weight
			self.conv.weight = nn.parameter.Parameter(self.ref_conv.weight.repeat(length, 1, 1)) # repeat copies data
			# self.convf = nn.Conv1d(length, length, 1)
			# self.convr = nn.Conv1d(length, length, 1)
		self.clm_mask = clm_mask
		self.expand_conv = expand_conv
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		print (x.shape)

		# for CLM training, apply lower triangular mask to convolution weights
		if self.clm_mask:
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

				# rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
				# # # softmax weights
				# # self.conv.weight.data = self.softmax(self.conv.weight.data)

				# mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				# applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
				# self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		else:
			# reshaped_conv = rearrange(self.conv.weight, 'f d p -> (p f) d', p=1)
			# masked_conv = torch.diagonal_scatter(reshaped_conv, torch.zeros(self.length-1), 1)
			# self.conv.weight.data = rearrange(masked_conv, '(p f) d -> f d p', p=1).contiguous()

			masked_convf = torch.tril(rearrange(self.convf.weight, 'f d p -> p f d'))
			self.convf.weight.data = rearrange(masked_convf, 'p f d -> f d p').contiguous()

			masked_convr = torch.triu(rearrange(self.convr.weight, 'f d p -> p f d'), diagonal=2)
			self.convr.weight.data = rearrange(masked_convr, 'p f d -> f d p').contiguous()

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
				clm_mask=True
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			 self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids.unsqueeze(1)
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_textbooks_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

tokenized_length = 512
dim = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 1).float().to(device)

# one = torch.tensor([[[1, 4, 3]]]).to(device)
# two = torch.tensor([[[1, 2, 3]]]).to(device)
# print (model(one, labels=one))
# print (model(two, labels=two))
# print (model)

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	print ()
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

# cached dataset
train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")


def tile_inputs(input_ids, tile_overlap=100, tile_size=828):
	text_length = len(input_ids[0])
	assert text_length > tile_overlap, 'Text must be longer than overlap to tile'
	tiled_arr = []
	i = 0
	while i < text_length:
		if i + tile_size <= text_length:
			tiled_arr.append(input_ids[0][i:i+tile_size])
		else:
			# pad the last tile to the appropriate length
			tokens = input_ids[0][i:i+tile_size]
			pad_length = tile_size - len(tokens)
			tokens = torch.nn.functional.pad(tokens,
											(0, pad_length),
											 mode='constant',
											 value=tokenizer.pad_token_id)
			tiled_arr.append(tokens)
		i += tile_size - tile_overlap
	return tiled_arr

def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output


def batch_tokenize_input(train_text, test_text, length=2000, batch_size=1024):
	train_data, test_data = [], []
	max_length = 512

	for i in range(0, length, batch_size):
		input_ids = tokenizer.batch_encode_plus(
			train_text[i:i+batch_size]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=max_length,
			padding='max_length'
		).input_ids
		train_data.append(input_ids)

	for i in range(0, len(test_text), batch_size):
		input_ids = tokenizer.batch_encode_plus(
			test_text[i:i+batch_size]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=max_length,
			padding='max_length'
		).input_ids
		test_data.append(input_ids)

	train_data = debatch_input(train_data)
	test_data = debatch_input(test_data)

	return train_data, test_data

def tokenize_input(train_text, test_text):
	train_data, test_data = [], []
	max_length = 512

	for i in range(1000000):
		input_ids = tokenizer.encode(
			train_text[i]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			max_length=max_length,
			padding='max_length'
		)

		if len(input_ids[0]) > max_length:
			input_set = tile_inputs(input_ids, tile_size=max_length)
			for inp in input_set:
				train_data.append(inp)
		else:
			train_data.append(input_ids)

	for i in range(len(test_text)):
		if test_text[i]:
			input_ids = tokenizer.encode(
				test_text[i]['text'],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=False,
				max_length=max_length,
				padding='max_length'
			)

			if len(input_ids[0]) > max_length:
				input_set = tile_inputs(
					input_ids,
					tile_size=max_length
				)
				for inp in input_set:
					test_data.append(inp)
			else:
				test_data.append(input_ids)

	return train_data, test_data

# train_data, test_data = batch_tokenize_input(train_text, valid_text)
# train_data, test_data = debatch_input(train_data), debatch_input(test_data)

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer modelz`
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


if isinstance(model, LlamaForCausalLM):
	reformat_inputs(train_data, test_data)


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_textbooks_8k")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='left'
n_vocab = len(tokenizer)

train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=True)
print ('dataset loaded')

def tokenization(example):
    tokens = tokenizer.batch_encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=512,
			padding='max_length',
		)
    return tokens

dataset = train_text.map(tokenization, batched=True)

mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=100,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=8e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_mixer_linear',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True,
	max_steps=1000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=dataset,
	eval_dataset=dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train() # '/home/bbadger/Desktop/tinystories_mixer_128_f_n8/checkpoint-748000'

for name, param in model.named_parameters():
	print (name)





