import torch
import torch.nn.functional as F
import datasets
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import json
import random
from accelerate import infer_auto_device_map
from safetensors.torch import load_model
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM

def generate_sample(query_dataset, target_dataset, index, dataset_size=20000, start_index=180000, n_context=128, replace=False):
	prob_weights = torch.ones(dataset_size)
	input = [query_dataset[index]]
	prob_weights[index-start_index] = 0
	indices = torch.multinomial(prob_weights, n_context-1, replacement=replace)
	for i in indices:
		target_text = reverse_tokenizer.decode(target_dataset[int(i)]['input_ids'])
		input.append(str(target_text))
	target_index = random.randint(1, n_context-1) # random index to put target embedding
	input[target_index] = reverse_tokenizer.decode(target_dataset[int(index)]['input_ids'])
	return input, target_index


def last_token_pool(last_hidden_states: Tensor,
				 attention_mask: Tensor) -> Tensor:
	left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
	if left_padding:
		return last_hidden_states[:, -1]
	else:
		sequence_lengths = attention_mask.sum(dim=1) - 1
		batch_size = last_hidden_states.shape[0]
		return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
	return f'Instruct: {task_description}\nQuery: {query}'


bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.float16
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct")
model = AutoModel.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct", quantization_config=bnb_config, device_map='auto')
reverse_tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")

def load_dataset(finemath=True):
	if not finemath:
		target_dataset = datasets.load_from_disk('/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512')
		query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_0_50000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_50000_100000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_100000_150000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_150000_200000.json'))]

	else:
		target_dataset = datasets.load_from_disk('/home/bbadger/Desktop/finemath-4-tokenized-train-c512-lpad-8k')
		query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_0_50000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_50000_100000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_100000_150000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_150000_200000.json'))]
	return query_dataset


query_dataset = load_dataset()
total_correct = 0
total = 0
for i in range(180000, 200000):
	# Each query must come with a one-sentence instruction that describes the task
	n_samples = 32
	task = 'Given a summary of a passage, find the corresponding text.'
	queries = [
		get_detailed_instruct(task, query_dataset[i])
	]
	# No need to add instruction for retrieval documents
	samples, target_index = generate_sample(query_dataset, target_dataset, i, n_context=n_samples)

	#samples[0] = str(queries[0])
	samples[0] = query_dataset[i]
	max_length = 512
	# Tokenize the input texts
	batch_dict = tokenizer(samples, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
	with torch.no_grad():
		outputs = model(**batch_dict)
		embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

		# normalize embeddings
		embeddings = F.normalize(embeddings, p=2, dim=1)
		scores = (embeddings[:1] @ embeddings[1:].T) * 100
		top_index = int(torch.topk(scores, 1).indices[0])
		print ('Top index, target index', top_index, target_index)
		if top_index+1 == target_index:
			total_correct += 1
		total += 1
		print (f'Top-1 accuracy: ', total_correct / total)

	
