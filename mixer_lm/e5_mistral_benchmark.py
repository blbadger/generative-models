import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct")
model = AutoModel.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct")
reverse_tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")

def generate_sample(self, query_dataset, target_dataset, index, dataset_size=20000, n_context=512, replace=False):
    self.n_context = n_context
    prob_weights = torch.ones(target_embeddings.shape[0])
    input = torch.zeros((n_context, query_embeddings[0].shape[1]))
    input[0] = query_dataset[index]
    prob_weights[index] = 0
    indices = torch.multinomial(prob_weights, n_context-1, replacement=replace)
    input[1:] = target_dataset[indices]
    target_index = random.randint(1, self.n_context-1) # random index to put target embedding
    matching_target = self.target_embeddings[index] # target the query matches
    input[target_index] = target_dataset[index]
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


total_correct = 0
total = 0
for i in range(180000, 200000):
    # Each query must come with a one-sentence instruction that describes the task
    target_dataset = datasets.load_from_disk('fineweb-edu-tokenized-train-c512')
    query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_0_50000.json'))]
    query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_50000_100000.json'))]
    query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_100000_150000.json'))]
    query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_150000_200000.json'))]
    n_sampes = 128
    task = 'Given a summary of a passage, find the corresponding text.'
    query = queries[i]
    queries = [
        get_detailed_instruct(task, query),
    ]
    # No need to add instruction for retrieval documents
    samples, target_index = generate_samples(query_dataset, target_dataset, i)
    for i in range(len(samples)):
        samples[i] = reverse_tokenizer.decode(samples[i])

    input_texts = queries + samples
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')

    max_length = 512
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    top_index = torch.topk(scores, 1)
    if top_index == 

    