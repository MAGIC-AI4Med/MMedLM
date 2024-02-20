import os
import numpy as np
import argparse
from transformers import AutoTokenizer
import multiprocessing as mp
from typing import List

class Tokenizer:
    """Class for handling the tokenization process."""

    def __init__(self, model_name_or_path: str, max_length: int = 2048):
        """Initialize the tokenizer with the specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.max_length = max_length
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def tokenize_and_save(self, filepaths: List[str], save_dir: str) -> None:
        """Tokenize the text files and save the tokenized content."""
        for filepath in filepaths:
            filename = os.path.basename(filepath).split(".")[0]
            with open(filepath, "r") as f:
                text = f.read()
            self._tokenize_text(text, filename, save_dir)

    def _tokenize_text(self, text: str, filename: str, save_dir: str) -> None:
        """Tokenize the text and handle the chunking and saving process."""
        tokenized_text = self.tokenizer.encode(text) + [self.eos_token_id]
        total_length = len(tokenized_text)
        cnt = 0
        if total_length < self.max_length:
            tokenized_text += [self.tokenizer.pad_token_id] * (self.max_length - total_length)
            np.save(os.path.join(save_dir, f"{filename}_{cnt}.npy"), np.array(tokenized_text))
        else:
            for i in range(0, total_length, 3 * self.max_length // 4):
                if i + self.max_length < total_length:
                    text_chunk = tokenized_text[i:i + self.max_length]
                else:
                    text_chunk = tokenized_text[-self.max_length:]
                np.save(os.path.join(save_dir, f"{filename}_{cnt}.npy"), np.array(text_chunk))
                if i + self.max_length >= total_length:
                    break
                cnt += 1


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Tokenize text files and save as NumPy arrays.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained model name or path for the tokenizer.')
    parser.add_argument('--train_filenames', type=str, required=True, help='Path to the a txt file containing paths of training files.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the tokenized files.')
    parser.add_argument('--cpu_num', type=int, default=16, help='Number of CPU processes to use for parallel processing.')
    return parser.parse_args()

def main() -> None:
    """Main function to handle the pre-tokenization of the dataset."""
    args = parse_args()
    print(f"Pre-tokenizing the dataset from {args.train_filenames}...")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(args.train_filenames, "r") as f:
        filepaths = [line.strip() for line in f.readlines()]

    tokenizer = Tokenizer(args.model_name_or_path)
    pool = mp.Pool(args.cpu_num)
    # Split the file list into chunks for each process
    file_chunks = [filepaths[i::args.cpu_num] for i in range(args.cpu_num)]
    for file_chunk in file_chunks:
        pool.apply_async(tokenizer.tokenize_and_save, args=(file_chunk, args.save_dir))
    pool.close()
    pool.join()
    print("Done! Total number of tokenized files:", len(os.listdir(args.save_dir)))

if __name__ == "__main__":
    main()
    
