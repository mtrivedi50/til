from pathlib import Path
import torch
import random


class CharacterDataset(torch.utils.data.Dataset):

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_dataset() -> tuple[list[str], dict[str, int]]:
    names_fpath = Path(__file__).parent / "names.txt"
    with open(names_fpath, "r") as f:
        names = f.read()
    names = names.split("\n")

    # Build vocabulary of characters, map idx -> char and vice-versa
    chars = sorted(list(set(''.join(names))))
    chars_to_i = {c: i+1 for i, c in enumerate(chars)}
    chars_to_i["."] = 0
    return names, chars_to_i


def _build_dataset(names: list[str], chars_to_i: dict[str, int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []
    for n in names:
        context = [0] * block_size
        for ch in n + ".":
            idx = chars_to_i[ch]
            X.append(context)
            y.append(idx)
            context = context[1:] + [idx]

    X = torch.tensor(X)
    y = torch.tensor(y)
    return X, y


def build_datasets(names: list[str], chars_to_i: dict[str, int], block_size: int, use_dataset_cls: bool = True) -> dict[str, CharacterDataset] | dict[str, tuple[torch.Tensor, torch.Tensor]]:
    random.seed(42)
    random.shuffle(names)
    train_size = int(0.8 * len(names))
    train_val_size = int(0.9 * len(names))

    Xtr, Ytr = _build_dataset(names[:train_size], chars_to_i, block_size)
    Xval, Yval = _build_dataset(names[train_size:train_val_size], chars_to_i, block_size)
    Xtest, Ytest = _build_dataset(names[train_val_size:], chars_to_i, block_size)

    if use_dataset_cls:
        return {
            "train": CharacterDataset(Xtr, Ytr),
            "val": CharacterDataset(Xval, Yval),
            "test": CharacterDataset(Xtest, Ytest)
        }
    else:
        return {
            "train": (Xtr, Ytr),
            "val": (Xval, Yval),
            "test": (Xtest, Ytest)
        }

