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
from transformers import LlamaConfig, LlamaForCausalLM
import prettytable
from prettytable import PrettyTable


dim = 128
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_heads': 16,
    'vocab_size': 4096
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)
print (model)

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

	for i in range(64000):
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

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer model
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


if isinstance(model, LlamaForCausalLM):
	reformat_inputs(train_data, test_data)


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=10,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=1000,
	save_steps=1000,
	learning_rate=1e-4,
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_llama',
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
model.train()
trainer.train()