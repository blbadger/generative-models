import os
import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM
import prettytable
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import datasets

device = 0 if torch.cuda.is_available else 'cpu'

dim = 512
context_length = 1024
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': 16,
	'num_attention_heads': 4,
	'vocab_size': 8000
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()

class MTPTransformer(nn.Module):

	def __init__(self, model, n_tokens):
		super().__init__()
		self.model = model
		self.n_tokens = 2

	def forward(self, x: torch.Tensor, labels=None, **kwargs):
		loss = torch.tensor([0], requires_grad=True)
		for i in range(n_tokens):
			output = self.lm_head(self.model(x)[0])
			shift_logits = output[..., :-(1 + i)].contiguous()
			shift_labels = labels[..., (1 + i):].contiguous()
			loss += self.cel(shift_logits, shift_labels)
			x = torch.argmax(model_output, dim=-2)

		return output, loss


model = MTPTransformer(model)
# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)


train_path = "/home/bbadger/Desktop/finemath-4-tokenized-train-c1024-8k"
test_path = "/home/bbadger/Desktop/finemath-4-tokenized-test-c1024-8k"

#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=8,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/mtp_finemath',
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
#trainer.train() 
trainer.train('/home/bbadger/Desktop/finemath_llama_n16_h4_c1024/checkpoint-60000')
