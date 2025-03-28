from datasets import load_dataset
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import os
from transformers import AutoTokenizer
import time
import torch
from transformers import AutoTokenizer, BatchEncoding



train_text = load_dataset("open-phi/textbooks", split="train")[:1600]["markdown"]
valid_text = load_dataset("open-phi/textbooks", split="train")[1600:]["markdown"]
print (len(train_text), len(valid_text))
print (train_text[0])

# file_path = "/home/bbadger/Desktop/TinyStories-train.txt"
# dataset = load_dataset("roneneldan/TinyStories")
dataset = load_dataset("open-phi/textbooks")
old_tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

class TextDataset(torch.utils.data.Dataset):
    """
    Create a Dataset object from a file consisting of lines of strings
    """
    def __init__(self, file_path, batch_size, truncation_index=2000000):
        super().__init__()
        self.batch_size = batch_size
        self.lines = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.lines.append(line.strip())  # Remove newline characters
        self.lines = self.lines[:truncation_index]

        self.line_batches = []
        batches = len(self.lines) // batch_size
        for i in range(batches):
            self.line_batches.append(self.lines[i*batch_size: i*(batch_size+1)])
        print (f'Batches to tokenize: {len(self.line_batches)}')

    def __len__(self):
        return len(self.line_batches)

    def __getitem__(self, idx):
        print (f"Tokenizing batch {idx}") if idx % 100 == 0 else None
        batch = self.line_batches[idx]
        return batch

# Create the dataset, and process the full file. 
# dataset = TextDataset(dataset, batch_size=1024)
# DataLoader for efficient batch processing
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

def get_training_corpus(dataset):
    dataset = dataset["train"]
    for i in range(len(dataset)):
        sample = dataset[i]
        print (len(sample['markdown']))
        yield sample['markdown']

training_corpus = get_training_corpus(dataset)
# print (next(training_corpus))

# Train the new tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(dataloader, 8192)
tokenizer.save_pretrained("/home/bbadger/Desktop/tiny_token_8k")
print ("Tokenizer saved")

# with open("/home/bbadger/Desktop/TinyStories-train.txt", "r") as file:
#     dataset = file.read()

# batch_size = 10000
# all_texts = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
# print (len(all_texts))

# def batch_iterator():
#     for i in range(0, len(dataset), batch_size):
#         yield dataset[i:i + batch_size]


# old_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# print ("pretrained tokenizer loaded")
# vocab_size = 5000
# data = iter(all_texts)
# print (next(data))
# tokenizer = old_tokenizer.train_new_from_iterator(data, vocab_size=vocab_size)
# print (f"new {vocab_size} size tokenizer trained")
# tokenizer.save_pretrained("/home/bbadger/Desktop/tiny_token_5k")



# paths = [str(Path("/home/bbadger/Desktop/TinyStories-train.txt"))]
# print (paths)

# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()

# # Customize training
# tokenizer.train(files=paths, vocab_size=5_000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])

# save_path = "/home/bbadger/Desktop/tiny_tokenizer"
# if not os.path.isdir(save_path):
# 	os.mkdir(save_path)

# # Save files to disk
# tokenizer.save_pretrained(save_path)
