"""
Run from root directory using:

uv run python -m ml.makemore.wavenet
"""

from pydantic import BaseModel, Field, model_validator
import math

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


class WavenetConfig(BaseModel):
    """
    Wavenet model: https://arxiv.org/pdf/1609.03499

    Follows dilated causal convolution pattern.
    """
    n_hidden: int = Field(
        description="Number of neurons in hidden layers. For simplicity, all hidden layers must have the same number of neurons."
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

    @model_validator(mode="after")
    def validate_context_length(self) -> "WavenetConfig":
        # context_len must be a power of two
        if math.log(self.context_len, 2) % 1 > 0:
            raise Exception("`context_len` must be a power of 2!")    
        return self


class LinearWithBatchNorm(nn.Module):

    def __init__(self, fanin: int, fanout: int):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.linear = nn.Linear(fanin, fanout, False)
        self.batch_norm = nn.BatchNorm1d(self.fanout)

    def forward(self, x: Tensor):
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(2, 1)
        return x


class Wavenet(nn.Module):

    def __init__(self, config: WavenetConfig):
        super().__init__()

        self.context_len = config.context_len

        # Embedding lookup table
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim)

        # Hidden layers. We're working with two batch dimensions. The first batch
        # dimension is the size of the training batch, the second batch dimension are
        # the concatenated pairs of feature embeddings. Note that we need to be careful
        # about how we move data between the linear layer and the BatchNorm1d layers.
        
        # Input Tensor: 64 x 8
        # Embedding:
        #   64 x 8 --> 64 x 8 x 10
        # First Hidden layer
        #   Flatten: 
        #       64 x 8 x 10 --> 64 x 4 x 20
        #   Linear: fanin=20, fanout=100
        #       64 x 4 x 20 --> 64 x 4 x 100
        #   BatchNorm1d: 100 features
        # Second Hidden Layer:
        #   FLatten:
        #       64 x 4 x 100 --> 64 x 2 x 200
        #   Linear: fanin=200, fanout=100
        #       64 x 2 x 200 --> 64 x 2 x 100
        #   BatchNorm: 100 features
        # Third Hidden Layer:
        #   FLatten:
        #       64 x 2 x 100 --> 64 x 1 x 200
        #   Linear: fanin=200, fanout=100
        #       64 x 1 x 200 --> 64 x 1 x 100
        #   BatchNorm: 100 features
        # Output
        #   FLatten:
        #       64 x 1 x 200 --> 64 x 100
        #   Linear: fanin=100, fanout=27
        #       64 x 100 --> 64 x 27

        self.layers = nn.ModuleList()
        n_hidden_layers = int(math.log(self.context_len, 2))
        for i in range(n_hidden_layers):
            cur = []
            # Special case: i == 0. The first hidden layer accepts the first round of
            # flattened embeddings.
            if i == 0:
                fanin = config.embedding_dim*2
            else:
                fanin = config.n_hidden*2
            
            # Batch norm vs. regular linear layer
            if config.batch_norm:
                cur.append(
                    LinearWithBatchNorm(fanin, config.n_hidden)
                )
            else:
                cur.append(nn.Linear(fanin, config.n_hidden))
            cur.append(nn.Tanh())
            self.layers.append(nn.Sequential(*cur))

        # Output layer
        output_layer = nn.Linear(
            in_features=config.n_hidden,
            out_features=config.vocab_size + 1,  # output prob. distribution over all characters in vocab
            bias=True,  # no batch norm in output layer
        )
        self.layers.append(output_layer)

    def forward(self, x: Tensor, targets: Tensor | None = None):
        # Each row of input is a series of indices representing context characters
        x = self.embedding(x.long())

        # Forward pass
        for i, layer in enumerate(self.layers):
            batch_size, context_len, embedding_len = x.shape

            # At output, context_len will be 1. Ignore.
            if i < len(self.layers)-1:
                x = x.reshape((batch_size, context_len // 2, embedding_len * 2))
            x = layer(x)

        # Since context_len is 1, our shape will be (B, 1, C). Flatten the first
        # dimension.
        logits = x.flatten(start_dim=1)
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
    config = WavenetConfig(
        n_hidden=68,
        context_len=CONTEXT_LENGTH,
        embedding_dim=10
    )
    model = Wavenet(config)
    print(f"Number of parameters: {count_parameters(model)}")
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

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

            # # Forward
            logits, loss = model(x, y)

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
