# PyTorch Dataset and DataLoader
import torch
import yaml
import tiktoken
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    batch_size: int
    # num_workers: int =0 # Default to 0 for local debugging

    @classmethod
    def friom_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # We only oput the data section for this class
        return cls(**config_dict["data"])


class GPTDataset(Dataset):
    def __init__(self, txt, block_size):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        # 1. Encode the entire text into token IDs
        self.token_ids = self.tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 2. We shift the text by 1 to creat (input, target) pairs
        # If block_size is 1024, we need 1025 tokens to create a full pair.
        print(f"Dataset initialied with {len(self.token_ids)}tokens.")

    def __len__(self):
        # We can create as many samples as there are tokens, minus the window size
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        # x is the input sequence
        # y is the target sequence (x shifted right by one )
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_dataloader(file_path, batch_size, block_size, shuffle=True):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = GPTDataset(text, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
