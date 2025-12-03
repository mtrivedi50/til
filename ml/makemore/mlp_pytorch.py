"""
Run from root directory using:

uv run python -m ml.makemore.mlp_pytorch
"""

from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ml.makemore.utils import (
    load_dataset,
    build_datasets
)


# Constants
BLOCK_SIZE = 8  # number of characters to use as context
ALPHABET_SIZE = 27  # size of alphabet
EMBEDDING_DIM = 10  # dimension for each embedding space (for each character)
N_HIDDEN = 100  # number of neurons in fully-connected layer


# Training data
names, chars_to_i = load_dataset()
all_datasets = build_datasets(names, chars_to_i, BLOCK_SIZE)
training_loader = DataLoader(dataset=all_datasets["train"], batch_size=64, shuffle=True)


# Model
model = nn.Sequential(
    nn.Embedding(num_embeddings=ALPHABET_SIZE, embedding_dim=EMBEDDING_DIM),
    nn.Flatten(),
    nn.Linear(BLOCK_SIZE * EMBEDDING_DIM, N_HIDDEN, bias=False), nn.BatchNorm1d(N_HIDDEN), nn.Tanh(),
    nn.Linear(                  N_HIDDEN, N_HIDDEN, bias=False), nn.BatchNorm1d(N_HIDDEN), nn.Tanh(),
    nn.Linear(                  N_HIDDEN, N_HIDDEN, bias=False), nn.BatchNorm1d(N_HIDDEN), nn.Tanh(),
    nn.Linear(                  N_HIDDEN, N_HIDDEN, bias=False), nn.BatchNorm1d(N_HIDDEN), nn.Tanh(),
    nn.Linear(                  N_HIDDEN, ALPHABET_SIZE)
)

optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

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
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        # Backward
        loss.backward()
        optimizer.step()

        # Store loss
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch}, batch {i+1}: average loss: {(running_loss / 100):.4f}")
            running_loss = 0

    avg_loss_epochs.append(epoch_loss / len(training_loader))


plt.plot(range(num_epochs), avg_loss_epochs)
plt.show()