import os
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
from safetensors.torch import save_file
from safetensors import safe_open

device = 0 if torch.cuda.is_available else 'cpu'

dim = 512 
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_attention_heads': 4,
    'vocab_size': 8000
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()

# gpt_config = transformers.OpenAIGPTConfig(vocab_size=4096, n_positions=512, n_embd=512, n_layer=8, n_head=4)
# model = transformers.OpenAIGPTLMHeadModel(gpt_config)

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_textbooks_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)
print (model)

# Causal mask check
# model = model.to(device)
# one = torch.tensor([[1, 2, 5]]).to(device)
# two = torch.tensor([[1, 2, 3]]).to(device)
# print (model(one, labels=one).logits)
# print (model(two, labels=two).logits)
# print (model)

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
train_text = load_dataset("open-phi/textbooks", split="train")[200:]['markdown']
valid_text = load_dataset("open-phi/textbooks", split="train")[:200]['markdown']

def tile_inputs(input_ids, tile_overlap=200, tile_size=1024):
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
	max_length = 1024

	for i in range(0, length, batch_size):
		input_ids = tokenizer.batch_encode_plus(
			train_text[i:i+batch_size],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=max_length,
			padding='max_length'
		).input_ids
		train_data.append(input_ids)

	for i in range(0, len(test_text), batch_size):
		input_ids = tokenizer.batch_encode_plus(
			test_text[i:i+batch_size],
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

	for i in range(len(train_text)):
		if i % 100 == 0: print (i)
		input_ids = tokenizer.encode(
			train_text[i],
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
				test_text[i],
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

tokenize_now=True
if tokenize_now:
	train_data, test_data = tokenize_input(train_text, valid_text)
	#train_data, test_data = debatch_input(train_data), debatch_input(test_data)

	data_dict = {
	'train_data': torch.stack(train_data, dim=0), 
	'test_data': torch.stack(test_data, dim=0)
	}

	save_file(data_dict, '/home/bbadger/Desktop/tokenized_textbook_8k.safetensors')
	print ('tokens saved')
else:
	tensors = {}
	with safe_open("/home/bbadger/Desktop/tokenized_textbook_8k.safetensors", framework="pt", device="cpu") as f:
		for key in f.keys():
			tensors[key] = f.get_tensor(key)

	train_data = list(tensors['train_data'])
	test_data = list(tensors['test_data'])
print (train_data[0].shape)

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
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/textbooks_llama_512_n8_h4_b32',
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
trainer.train() # '/home/bbadger/Desktop/tinystories_llama_256/checkpoint-96000'
# trainer.train('/home/bbadger/Desktop/tinystories_autollama_512_n8/checkpoint-164000') # '/home/bbadger/Desktop/tinystories_autollama_512_n8/checkpoint-284000'