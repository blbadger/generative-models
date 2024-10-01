import os
from enum import Enum
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

def create_and_prepare_model(args, data_args, training_args):
	bnb_config = None
	quant_storage_dtype = None

	if args.use_4bit_quantization:
		compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
		quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

		bnb_config = BitsAndBytesConfig(
			load_in_4bit=args.use_4bit_quantization,
			bnb_4bit_quant_type = args.bnb_4bit_quant_type,
			bnb_4bit_compute_dtype = compute_dtype,
			bnb_4bit_use_double_quant = args.use_nested_quant,
			bnb_4bit_quant_storage = quant_storage_dtype
			)

		if args.use_8bit_quantization:
			bnb_config = BitsAndBytesConfig(load_in_8bit=True)

	torch_dtype = quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
	model = AutoModelForCausalLM.from_pretrained(
			args.model_name_or_path,
			quantization_config = bnb_config,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
			torch_dtype=torch_dtype
		)

	peft_config=None
	if args.use_peft_lora:
		peft_config = LoraConfig(
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			r=args.lora_r,
			bias="none",
			task_type="CAUSAL_LM",
			target_modules=args.lora_target_modules.split(",")
			if args.lora_target_modules != "all-linear"
			else args.lora_target_modules
			)

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