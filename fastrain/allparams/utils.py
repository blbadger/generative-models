import os
from enum import Enum
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

def create_and_prepare_model(args, data_args, training_args):
	peft_config = None
	torch_dtype = quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
	model = AutoModelForCausalLM.from_pretrained(
			args.model_name_or_path,
			quantization_config = bnb_config,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
			torch_dtype=torch_dtype
		)
	
	# TODO: add special token compatibility
	special_tokens=None
	if special_tokens is not None:
		tokenizer = AutoTokenizer.from_pretrained(
			args.model_name_or_path,
			pad_token=special_tokens.pad_token.value,
			bos_token=special_tokens.bos_token.value,
			eos_token=special_tokens.eos_token.value,
			additional_special_tokens=special_tokens.list(),
			trust_remote_code=True
			)
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
		tokenizer.pad_token = tokenizer.eos_token

	return model, peft_config, tokenizer