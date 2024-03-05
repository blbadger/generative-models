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

	def __init__(self, n_vocab, dim, depth, mixer_mask=True):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				mixer_mask = mixer_mask
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

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
# model = AutoModel.from_pretrained('/home/bbadger/Desktop/tinystories_mixer/checkpoint-15000')
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

# barebones MLP mixer, expects an embedding on input tokens
tokenized_length = 512
dim = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 8, mixer_mask=False).float().to(device)
load_model(model, '/home/bbadger/Desktop/tinystories_mixer/checkpoint-15000/model.safetensors')
prompt = 'Once upon a time there was a tree named Barky. Barky liked the'
tokens = tokenizer.encode(
				prompt,
				add_special_tokens=False,
				return_tensors='pt',
				padding=True,
			)
for i in range(20):
	output = model(tokens)
	print (output.shape)
