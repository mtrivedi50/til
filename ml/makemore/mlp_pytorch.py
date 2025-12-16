"""
Run from root directory using:

uv run python -m ml.makemore.mlp_pytorch
"""

from pydantic import BaseModel, Field, model_validator

from torch import nn, Tensor
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Any

from ml.makemore.utils import (
    load_dataset,
    build_datasets
)


class MLPConfig(BaseModel):
    n_hidden_layers: int = Field(
        description="Number of hidden layers."
    )
    n_hidden: list[int] = Field(
        description=" ".join([
            "Number of neurons in hidden layers. If a list, then the length must be equal to `n_hidden_layers`.",
            "If constant, then all hidden layers will have the same number of neurons."
        ])
    )
    context_len: int = Field(
        description="Number of characters to use as context."
    )
    vocab_size: int = Field(
        description="Number of characters in the vocabulary.",
        default=26
    )
    embedding_dim: int = Field(
        description="Size of character embedding."
    )
    bias: bool = Field(
        description="Learn bias for each linear layer. Default is False, because our default behavior is to add batch norm layers.",
        default=False
    )
    batch_norm: bool = Field(
        description="Add batch norm layers after each linear layer. Default is True.",
        default=True
    )

    @model_validator(mode="before")
    @classmethod
    def convert_n_hidden_to_list(cls, data: dict[str, Any]):
        if isinstance(data, dict):
            flag_n_hidden = "n_hidden" in data and isinstance(data["n_hidden"], int)
            flag_n_hidden_layers = "n_hidden_layers" in data and isinstance(data["n_hidden_layers"], int)
            if flag_n_hidden and flag_n_hidden_layers:
                n_hidden_list = [data["n_hidden"]] * data["n_hidden_layers"]
                data["n_hidden"] = n_hidden_list
        return data

    @model_validator(mode="after")
    def validate(self) -> "MLPConfig":
        if isinstance(self.n_hidden, list):
            if len(self.n_hidden) == 0:
                raise ValueError("n_hidden cannot be an empty list!")
            if len(self.n_hidden) != self.n_hidden_layers:
                raise ValueError("n_hidden is a list, but its length does not equal `n_hidden_layers`!")
        return self


class MLP(nn.Module):

    def __init__(self, config: MLPConfig):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for i in range(config.n_hidden_layers-1):
            if i == 0:
                fanin = config.context_len * config.embedding_dim
            else:
                fanin = config.n_hidden[i]
            
            self.layers.append(
                nn.Linear(fanin, config.n_hidden[i+1], config.bias), 
            )
            if config.batch_norm:
                self.layers.append(nn.BatchNorm1d(config.n_hidden[i+1]))

        # Output layer
        last_hidden_layer_fanin = config.n_hidden[-1]
        self.output_layer = nn.Linear(
            in_features=last_hidden_layer_fanin,
            out_features=config.vocab_size + 1,  # output prob. distribution over all characters in vocab
            bias=True,  # no batch norm in output layer
        )

    def forward(self, x: Tensor, targets: Tensor | None = None):
        # Each row of input is a series of indices representing context characters
        x = self.embedding(x.int()).view(x.size()[0], -1)
        for layer in self.layers:
            x = layer(x)
        logits = self.output_layer(x)
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss


if __name__ == "__main__":
    # Use 8 characters for context length
    CONTEXT_LENGTH = 8

    # Training data
    names, chars_to_i = load_dataset()
    all_datasets = build_datasets(names, chars_to_i, CONTEXT_LENGTH)
    training_loader = DataLoader(dataset=all_datasets["train"], batch_size=64, shuffle=True)

    # Model
    config = MLPConfig(
        n_hidden_layers=4,
        n_hidden=100,
        context_len=CONTEXT_LENGTH,
        embedding_dim=10
    )
    mlp = MLP(config)
    optimizer = optim.Adam(params=mlp.parameters(), lr=1e-3)

    # Training
    num_epochs = 5
    avg_loss_epochs = []
    for epoch in range(num_epochs):
        running_loss = 0
        epoch_loss = 0

        # Batch. As opposed to manually defining batches, use DataLoader class. This also
        # shuffles the rows for each epoch.
        for i, (x, y) in enumerate(training_loader):
            optimizer.zero_grad()

            # Forward
            logits, loss = mlp(x, y)

            # Backward
            loss.backward()
            optimizer.step()

            # Loss statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch}, batch {i+1}: average loss: {(running_loss / 100):.4f}")
                running_loss = 0

        avg_loss_epochs.append(epoch_loss / len(training_loader))

    # Loss plot
    plt.plot(range(num_epochs), avg_loss_epochs)
    plt.show()
