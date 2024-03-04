from datasets import load_dataset

# dataset = load_dataset("roneneldan/TinyStories")

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import os


paths = [str(Path("/home/bbadger/Desktop/TinyStories-train.txt"))]
print (paths)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=5_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

save_path = "/home/bbadger/Desktop/tiny_tokenizer"
if not os.path.isdir(save_path):
	os.mkdir(save_path)

# Save files to disk
tokenizer.save_pretrained(save_path)