import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import torch.nn as nn
from datasets import load_dataset, load_from_disk, Dataset
import sentencepiece
import json

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_16k")
tokenizer.pad_token = tokenizer.eos_token

def tokenization(example, n_ctx=128):
    tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			padding=True,
			padding_side='left'
		).input_ids
    return {'input_ids': tokens}

def map_dataset(path):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)
	dataset = text.map(tokenization, batched=False)
	dataset.save_to_disk(path)
	print ('Dataset saved to disk')
	return

query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_0_50000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_50000_100000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_100000_150000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_150000_200000.json'))]

path = "/home/bbadger/Desktop/contrastive-summaries-fineweb-rpad-200k"
map_dataset(path)




















