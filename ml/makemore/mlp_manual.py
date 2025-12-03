"""
Run from root directory using:

uv run python -m ml.makemore.mlp_manual
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ml.makemore.utils import (
    load_dataset,
    build_datasets
)


# Constants
BLOCK_SIZE = 3  # number of characters to use as context
ALPHABET_SIZE = 27  # size of alphabet
EMBEDDING_SIZE = 10  # dimension for each embedding space (for each character)
N_HIDDEN = 100  # number of neurons in fully-connected layer
LEARNING_RATE = 0.1  # size of step to take in gradient
MINIBATCH_SIZE = 32  # number of minibatches?
N = 1000  # training iterations


class Module:
    out: Tensor


class Linear(Module):
    fan_in: int
    fan_out: int
    bias: bool

    W: Tensor
    b: Tensor | None
    
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        self.fan_in = fan_in
        self.fan_out = fan_out

        # Random weights. Divide weights by sqrt(fan_in) to normalize variance of
        # weights and prevent activations from exploding or vanishing as they flow
        # through the network.
        self.W = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.b = torch.zeros(fan_out) if bias else None

    def __call__(self, x: Tensor):
        self.out = x @ self.W
        if self.b is not None:
            self.out += self.b
        return self.out

    @property
    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class BatchNorm1d(Module):
    dim: int
    eps: float
    track_running_stats: bool
    momentum: float

    training: bool
    gamma: Tensor
    beta: Tensor
    running_mean: Tensor
    running_std: Tensor

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """
        Method described in the paper Batch Normalization: Accelerating Deep Network
        Training by Reducing Internal Covariate Shift .
        """
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Parameters (trained via backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Running stats for computing batch mean/std dev during training.
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: Tensor):
        if self.training:
            x_mean = x.mean(0, keepdim=True)  # mean for each column
            x_var = x.var(0, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        self.out = ((x - x_mean) / torch.sqrt(x_var + self.eps)) * self.gamma + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

        return self.out
    
    @property
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Module):

    def __call__(self, x: Tensor):
        self.out = torch.tanh(x)
        return self.out
    
    @property
    def paramters(self):
        return []


if __name__ == "__main__":
    # Dataset
    names, chars_to_i = load_dataset()
    datasets = build_datasets(names, chars_to_i, block_size=BLOCK_SIZE, use_dataset_cls=False)
    Xtr, Ytr = datasets["train"]

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((ALPHABET_SIZE, EMBEDDING_SIZE), generator=g)
    layers: list[Module] = [
        Linear(BLOCK_SIZE * EMBEDDING_SIZE, N_HIDDEN), Tanh(),
        Linear(                   N_HIDDEN, N_HIDDEN), Tanh(),
        Linear(                   N_HIDDEN, N_HIDDEN), Tanh(),
        Linear(                   N_HIDDEN, N_HIDDEN), Tanh(),
        Linear(                   N_HIDDEN, N_HIDDEN), Tanh(),
        Linear(                   N_HIDDEN, ALPHABET_SIZE)
    ]
    with torch.no_grad():
        # Last layer less confident
        layers[-1].W *= 0.1
        # Since using Tanh, apply gain to all other layers. This is called Kaiming
        # Initialization.
        for l in layers[:-1]:
            l.W *= 5/3
    
    # All parameters
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True
    
    for i in range(N):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (MINIBATCH_SIZE,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass
        emb = C[Xb]  # embed chars into vectors
        x = emb.view(-1, BLOCK_SIZE * EMBEDDING_SIZE)  # concatenate character embeddings
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb)

        # backward pass
        for layer in layers:
            layer.out.retain_grad()
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < N/2 else 0.01
        for p in parameters:
            p.data += -lr * p.grad
        

