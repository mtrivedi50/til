from pathlib import Path
import torch
from torch.utils.data import Dataset
import shutil
import subprocess
from typing import Literal


START_TOKEN = "<START>"


def load_data(wkdir: Path, subdir: Path) -> str:
    """
    Load Shakespeare dataset from Karpathy's GitHub.
    """
    if not (subdir / "input.txt").exists():
        subprocess.run([
            "wget",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        ])
        shutil.move(wkdir / "input.txt", subdir / "input.txt")

    with open(subdir / "input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Add the start token to the beginning
    text = START_TOKEN + text

    return text

    
def define_alphabet(text: str) -> list[str]:
    """
    Process the input text. In particular, return:
        - A list of unique characters
        - A dictionary mapping characters to their associated index
    """
    unique_chars = sorted(list(set(text)))

    # Add a special "START" token
    unique_chars = [START_TOKEN] + unique_chars

    return unique_chars


def encode(text: str, chars_to_i: str) -> list[int]:
    return [chars_to_i[c] for c in text]


class CharacterLMDataset(Dataset):

    def __init__(self, data: torch.Tensor, context_len: int):
        if len(data.shape) > 1 or data.dtype != torch.long:
            raise Exception(f"Data should be a tensor of integer indices. Got {data.dim()} dimensions with type {data.dtype}.")
        self.data = data
        self.context_len = context_len

    def __len__(self):
        # Subtract 1 so that any sequence of characters definitely has a label.
        return len(self.data) - self.context_len - 1
    
    def __getitem__(self, index: int):
        # x -> (1, context_len)
        # y -< (1, context_len)
        # In batch, these will be (B, context_len)
        x_start, x_end = index, index + self.context_len
        y_start, y_end = index + 1, index + 1 + self.context_len
        x = self.data[x_start:x_end]
        y = self.data[y_start:y_end]
        return x, y


def build_dataset(
    text: str,
    chars_to_i: dict[str, int],
    train_size: float = 0.8,
    val_size: float = 0.1,
    context_len: int = 8,
) -> dict[Literal["train", "val", "test"], CharacterLMDataset]:
    data = torch.tensor(encode(text, chars_to_i), dtype=torch.long)

    if train_size + val_size > 1:
        raise Exception("Training and validation sizes exceed 100%!")

    # Split into training, validation, and test.
    train_idx = int(len(data) * train_size)
    val_idx = train_idx + int(len(data) * val_size)
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]

    # Build dataset classes
    return {
        "train": CharacterLMDataset(train_data, context_len),
        "val": CharacterLMDataset(val_data, context_len),
        "test": CharacterLMDataset(test_data, context_len),
    }
