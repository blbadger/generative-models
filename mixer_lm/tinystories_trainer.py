import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import einops
from einops import rearrange
import transformers
from transformers import TextDataset, Trainer, TrainingArguments
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset


def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expansion_factor=4, dropout=0.):
		super().__init__()
		self.layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(length)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim, expansion_factor=expansion_factor)
		self.conv = nn.Conv1d(dim, dim, 1)

		# for CLM training: mask conv weights to become upper-triangular
		if mixer_mask:
			self.conv.weight = torch.nn.Parameter(torch.triu(self.conv.weight))

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		x = rearrange(x, 'b t f -> b f t')
		residual = x
		x = self.conv(x) + residual
		x = self.seq_layernorm(x)
		x = rearrange(x, 'b f t -> b t f')
		residual = x
		x = self.patch_ff(x) + residual
		x = self.layernorm(x)
		return x

class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab)
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

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

# barebones MLP mixer, expects an embedding on input tokens
tokenized_length = 512
dim = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 8).float().to(device)
print (model)

import prettytable
from prettytable import PrettyTable

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
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].squeeze(0)
	return input_data

def tokenize_input(train_text, test_text):
	train_data, test_data = [], []
	max_length = 512

	for i in range(50000):
		input_ids = tokenizer.encode(
			train_text[i]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			max_length=max_length,
			padding='max_length'
		)

		if len(input_ids[0]) > max_length:
			pass
			# input_set = tile_inputs(input_ids, tile_size=max_length)
			# for inp in input_set:
			# 	train_data.append(inp)
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
				pass
				# input_set = tile_inputs(
				# 	input_ids,
				# 	tile_size=max_length
				# )
				# for inp in input_set:
				# 	test_data.append(inp)
			else:
				test_data.append(input_ids)

	return train_data, test_data

train_data, test_data = tokenize_input(train_text, valid_text)
# train_data, test_data = debetach_input(train_data), debatch_input(test_data)

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=1,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=64,
	warmup_steps=100,
	eval_steps=2000,
	save_steps=500,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_mixer',
	optim='adamw_torch',
	overwrite_output_dir=True,
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_data,
	eval_dataset=test_data,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()