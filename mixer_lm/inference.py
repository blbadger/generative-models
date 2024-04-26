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
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file



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

	def __init__(self, dim, length, mixer_mask=True, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1)
		self.mixer_mask = mixer_mask
		self.expand_conv = expand_conv

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.mixer_mask:
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
				rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
				self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

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
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			 self.wte.weight = self.lm_head.weight
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
		labels = labels.to(device)
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits[..., -100:], shift_labels[..., -100:])
		print (loss)
		return loss, output



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

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

train_data, test_data = batch_tokenize_input(train_text, valid_text)
# train_data, test_data = debatch_input(train_data), debatch_input(test_data)
n_vocab = len(tokenizer)

# barebones MLP mixer, expects an embedding on input tokens
tokenized_length = 512
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 8).float().to(device)

# model.load_state_dict(torch.load('/home/bbadger/Desktop/tinystories_mixer_512_flat/checkpoint-424000/pytorch_model.bin'))

load_model(model, '/home/bbadger/Desktop/tinystories_mixencode_512_f_n8/checkpoint-400000/model.safetensors')

# training_arguments = transformers.TrainingArguments(
# 	num_train_epochs=0,
# 	per_device_train_batch_size=16,
# 	per_device_eval_batch_size=1,
# 	warmup_steps=0,
# 	eval_steps=10,
# 	save_steps=200,
# 	learning_rate=2e-4,
# 	fp16=True, 
# 	evaluation_strategy='steps',
# 	output_dir='~/Desktop/tinystories_mixer_0',
# 	optim='adamw_torch',
# 	overwrite_output_dir=True,
# 	save_safetensors=False
# )

# trainer = transformers.Trainer(
# 	model=model,
# 	train_dataset=train_data,
# 	eval_dataset=test_data,
# 	args=training_arguments,
# 	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )

# # prompt = 'Once upon a time, there was a big dog named Barky. He wagged his tail and began to'

# model.train()
# trainer.train('/home/bbadger/Desktop/tinystories_mixer/checkpoint-700000')
model.eval()
tokens = test_data[5]

prompt = '''

'''
# tokens = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', max_length=512)

# tokens = tokenizer.encode(
# 				prompt,
# 				add_special_tokens=False,
# 				return_tensors='pt',
# 				padding='max_length',
# 				max_length=512
# 			)

print ('model loaded.')
print ('Input: ', tokenizer.decode(tokens[0]))
tokens = rearrange(tokens, '(b p) t -> b p t', p=1)

fout = []
for i in range(50, 1, -1):
	loss, output = model(tokens, labels=tokens)
	out_token = torch.topk(output, dim=1, k=1).indices.flatten()[-i]
	tokens[..., -i+1] = out_token

print ('\n \n')
print ('Output: \n', tokenizer.decode(tokens[0][0]))
