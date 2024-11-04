import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import torch.nn as nn
from datasets import load_dataset, load_from_disk, Dataset
import sentencepiece
import pyarrow as pa

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

def tokenization(example, n_ctx=32):
    tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False,
		).input_ids
    tokens = torch.flatten(tokens, start_dim=0)
    batch_size = len(tokens) // n_ctx #if len(tokens) % n_ctx==0 else len(tokens) // n_ctx + 1
    length = n_ctx * batch_size
    #tokens = tokenizer.pad(tokens, padding='max_length', max_length=length, padding_side='right')
    tokens = tokens[:length].reshape(batch_size, n_ctx)
    return {'input_ids': tokens}

train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c32-packed"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c32-packed"

def map_dataset(train_path, test_path, split_index=50000):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).skip(split_index)
	test_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).take(split_index)

	train_dataset = train_text.map(tokenization, batched=False)
	test_dataset = test_text.map(tokenization, batched=False)
	train_dataset.save_to_disk(train_path)
	test_dataset.save_to_disk(test_path)
	print ('Datasets saved to disk')
	return


def debatch(example):
	batch_size = len(example['input_ids'])
	keys = list(example.keys())
	for key in keys:
		if key != 'input_ids':
			example.pop(key, None)
	debatched_inputs = [{'input_ids': tokens} for tokens in example["input_ids"][0]]
	return pa.Table.from_pylist(debatched_inputs)

#map_dataset(train_path, test_path)
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
print (test_dataset[0]['input_ids'])
test_dataset = test_dataset.map(debatch, batched=True, batch_size=1)
print (test_dataset[0])
test_dataset.save_to_disk(test_path+'-debatched')
train_dataset = train_dataset.map(debatch, batched=True, batch_size=1)
print (train_dataset[0])
train_dataset.save_to_disk(train_path+'-debatched')
























