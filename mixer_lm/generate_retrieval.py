from llama_cpp import Llama
import json

model = Llama(
	model_path = '/home/bbadger/Desktop/models/llama-3-8b-instruct-Q8_0.gguf',
	n_gpu_layers = -1,
	chat_format='llama-3'
	)

# train/validation set: first 100k train examples, first 10k validation examples
train_text = load_dataset("roneneldan/TinyStories", split="train")[:100000]
valid_text = load_dataset("roneneldan/TinyStories", split="validation")[:10000]

batch_size = 16
outputs = []
for i in range(0, len(train_text), batch_size):
	output = llm.create_chat_completion(
	      messages = [
	          {"role": "system", "content": "You are an assistant for creating summaries for short stories."},
	          {
	              "role": "user",
	              "content": f"Give a brief one-sentence summary of the following story: {train_text[j]}"
	          }
	      for j in range(j, j + batch_size)
	      ]
	)
	output.append(output)

with open('output.txt', 'w') as f:
    f.write(output)




