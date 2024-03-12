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
from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as F


rotary_emb = RotaryEmbedding(dim = 32).to(0)

class PhaseAmplitudeRelu(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, z):
		return F.relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))

def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim).to(torch.cfloat),
		PhaseAmplitudeRelu(),
		nn.Linear(inner_dim, dim).to(torch.cfloat)
	)

def ConvForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1).to(torch.cfloat),
		PhaseAmplitudeRelu(),
		nn.Conv1d(inner_dim, dim, 1).to(torch.cfloat)
		)

class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expand_conv=True):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.expand_conv = expand_conv
		if self.expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1).to(torch.cfloat)
		
		# for CLM training, apply lower triangular mask to convolution weights
		self.mixer_mask = mixer_mask

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		if self.mixer_mask:
			if self.expand_conv:
				masked_conv0 = nn.Parameter(rearrange(torch.tril(rearrange(self.conv[0].weight, 'f d p -> f (d p)')), 'f (d p) -> f d p', p=1))
				masked_conv2 = nn.Parameter(rearrange(torch.tril(rearrange(self.conv[-1].weight, 'f d p -> f (d p)')), 'f (d p) -> f d p', p=1))
				self.conv[0].weight = masked_conv0
				self.conv[-1].weight = masked_conv2
			else:
				self.conv.weight = torch.nn.Parameter(rearrange(self.conv.weight, 'f d p -> f (d p)'))
				self.conv.weight = torch.nn.Parameter(torch.tril(self.conv.weight))
				self.conv.weight = torch.nn.Parameter(rearrange(self.conv.weight, 'f (d p) -> f d p', p=1))
		residual = x
		# x.real = self.seq_layernorm(x.real)
		x = self.conv(x) + residual
		residual = x
		# x.real = self.patch_layernorm(x.real)
		x = self.patch_ff(x) + residual
		return x

class LanguageMixer(nn.Module):

	def __init__(self, tokenized_length, n_vocab, dim, depth, tie_weights=False, complex_position=True):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False).to(torch.cfloat)
		if tie_weights:
			self.lm_head.weight = self.wte.weight
		self.cel = nn.CrossEntropyLoss()
		complex_position = torch.zeros(tokenized_length, dim)
		complex_position = complex_position.to(torch.cfloat)

		# positional encoding
		scale = 2 / tokenized_length
		for i in range(tokenized_length):
			complex_position[i, :] = 0 + scale*1j * i
		self.complex_position = complex_position.to(device)

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		x = x.to(torch.cfloat)

		for block in self.mixerblocks:
			# apply positional encoding
			x[..., :, :] += self.complex_position
			x = block(x)

		output = self.lm_head(x).to(torch.float)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

tokenized_length = 512
dim = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(tokenized_length, n_vocab, dim, 4)


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

def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output


def batch_tokenize_input(train_text, test_text, length=100000, batch_size=1024):
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


train_data, test_data = batch_tokenize_input(train_text, valid_text)
train_data, test_data = debatch_input(train_data), debatch_input(test_data)

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer modelz`
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=64,
	warmup_steps=500,
	eval_steps=1000,
	save_steps=1000,
	learning_rate=5e-4,
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_complexmixer',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=False
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_data,
	eval_dataset=test_data,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train()










