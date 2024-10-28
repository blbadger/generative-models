from llama_cpp import Llama
import json
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

# Instantiate the parser
parser = argparse.ArgumentParser(description='Driver args')
parser.add_argument('--start', type=int)
parser.add_argument('--stop', type=int)
parser.add_argument('--output_path', type=str)

if __name__ == '__main__':
	args = parser.parse_args()
	model = Llama(
		model_path = '/home/bbadger/Desktop/models/llama-3-8b-instruct-Q8_0.gguf',
		n_gpu_layers = -1,
		chat_format='llama-3',
		verbose=False,
		n_ctx=4096
		)

	train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train"
	train_text = load_from_disk(train_path)
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token

	outputs = []
	for j in tqdm(range(args.start, args.stop)):
		text = tokenizer.decode(train_text[j]['input_ids'])
		output = model.create_chat_completion(
		    messages = [
				{"role": "system", "content": "You are a helpful assistant giving summaries of text for future searches."},
					{
					"role": "user",
					"content": f"Give a brief one-sentence summary of the following text and return no other text: {text}"
					}
			]
		)
		outputs.append(output)

	output_path = args.output_path + f'_{start}_{stop}.json'
	with open(args.output_path, 'w') as f:
	    json.dump(outputs, f)
