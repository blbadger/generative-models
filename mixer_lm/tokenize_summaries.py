import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import torch.nn as nn
from datasets import load_dataset, load_from_disk, Dataset
import sentencepiece
import json

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

def tokenization(example, n_ctx=128):
    tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=512,
			padding=True,
			padding_side='left'
		).input_ids
    return {'input_ids': tokens}

def map_dataset(array, label='summary'):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	tokenized_array = []
	count = 0 
	for sample in array:
		tokens = tokenizer.encode_plus(
			sample,
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=512,
			padding='max_length',
			padding_side='left'
		).input_ids
		tokenized_array.append(tokens[0])
	output_dict = {label: torch.tensor(tokenized_array)}
	return output_dict

def extract_tokens(dataset, limit=200000, label='text'):
	array = []
	count = 0
	idata = iter(dataset)
	for sample in dataset:
		count += 1
		if count > limit:
			break
		array.append(next(idata)['input_ids'])
	output_dict = {label: torch.tensor(array)}
	return output_dict


query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_0_50000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_50000_100000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_100000_150000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_150000_200000.json'))]

path = "/home/bbadger/Desktop/constrastive-fineweb-lpad-200k.safetensors"
summary_datset = map_dataset(query_text, label='summary')

summary_path = "/home/bbadger/Desktop/contrastive-summaries-fineweb-lpad-200k"
summary_tokens = load_from_disk(summary_path, keep_in_memory=None)
text_dataset = map_dataset(text, label='text')
datset = summary_dataset + text_dataset
save_file(dataset, path)





















