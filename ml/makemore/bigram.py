"""
Bigram-level character model. Run from root directory using:

uv run python -m ml.bigram
"""

from pydantic import BaseModel, Field, model_validator
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ml.makemore.utils import (
    load_dataset,
    build_datasets,
    count_parameters,
)


class Bigram(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: int, target: int | None = None):
        # x = (B, C)
        # B = batch size
        # C = channels / features (size of character embedding)
        x = self.embedding(idx)

        # Embeddings themselves can be considered logits
        loss = None
        if target is not None:
            loss = F.cross_entropy(x, torch.Tensor(target))
        return x, loss


if __name__ == "__main__":
    # Names dataset
    names, chars_to_i = load_dataset()

    # Training / validation dataset
    train_size = int(0.8 * len(names))
    train_val_size = int(0.9 * len(names))
    names_tr = names[:train_size]
    names_val = names[train_size:train_val_size]
    
    # Bigram model requires pairs of sequential characters
    def _convert_name_to_bigram(name: str, chars_to_i: dict[str, int]) -> tuple[list[str], list[str]]:
        x, y = [], []
        for i, j in zip("." + name, name + "."):
            x.append(chars_to_i[i])
            y.append(chars_to_i[j])
        return x, y

    def convert_names_to_bigrams(names: list[str], chars_to_i: dict[str, int]) -> tuple[Tensor, Tensor]:
        X, y = [], []
        for n in names:
            curX, curY = _convert_name_to_bigram(n, chars_to_i)
            X.extend(curX)
            y.extend(curY)
        return torch.tensor(X), torch.tensor(y)

    Xtr, ytr = convert_names_to_bigrams(names_tr, chars_to_i)
    Xval, yval = convert_names_to_bigrams(names_val, chars_to_i)
    model = Bigram(len(chars_to_i))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    training_losses = []
    val_losses = []
    n = 10000
    batch_size = 32
    for i in range(n):
        optimizer.zero_grad()
        
        # Batch
        batch = torch.randint(low=0, high=train_size, size=(batch_size,))
        Xb, yb = Xtr[batch], ytr[batch]

        # Forward & backward pass
        logits, loss = model(Xb, yb)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Step {i}: training loss = {loss.item():4f}")

        # Loss statistics. Note: since we are plotting the loss for every batch, this is
        # going to be super jagged.
        training_losses.append(loss.item())
        val_losses.append(model(Xval, yval)[1].item())
    
    plt.plot(range(n), training_losses, color="blue", label="training")
    plt.plot(range(n), val_losses, color="red", label="validation")
    plt.show()