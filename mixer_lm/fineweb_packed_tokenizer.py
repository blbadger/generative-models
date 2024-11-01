import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
from datasets import load_dataset, load_from_disk
import sentencepiece
from tokenizers import ByteLevelBPETokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

def tokenization(example, n_ctx=32):
    tokens = tokenizer.batch_encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False,
		).input_ids
    tokens = torch.flatten(start_dim=0)
    batch_size = len(tokens) // n_ctx if len(tokens) % n_ctx==0 else len(tokens) // n_ctx + 1
    length = n_ctx * batch_size
    tokens = tokenizer.pad(tokens, padding='max_length', max_length=length, padding_side='right')
    tokens = tokens.reshape(batch_size, n_ctx)
    return tokens

train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c32-packed"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c32-packed"

def map_dataset(train_path, test_path, split_index=50000):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).skip(split_index)
	test_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).take(split_index)

	train_dataset = train_text.map(tokenization, batched=True)
	test_dataset = test_text.map(tokenization, batched=True)
	train_dataset.save_to_disk(train_path)
	test_dataset.save_to_disk(test_path)
	print ('Datasets saved to disk')
	return

map_dataset(train_path, test_path)
