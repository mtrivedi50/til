from pydantic import BaseModel, Field
import torch
from torch import nn
import torch.nn.functional as F


class TinyGptConfig(BaseModel):
    vocab_size: int = Field(
        description="Total size of vocab."
    )
    block_size: int = Field(
        description="Maximum context length for predictions.",
        default=256
    )
    n_embd: int = Field(
        description="Size of embedding used to represent token.",
        default=32,
    )
    n_head: int = Field(
        description="Number of single-attention heads to concatenate to create the multi-attention head.",
        ge=2,
    )
    n_layer: int = Field(
        description=" ".join([
            "The number of layers to use within the decoder.",
            "An individual layer is a masked multi-attention head followed by a feed-forward MLP."
        ]),
        ge=1,
    )
    ffw_inner_scale: int = Field(
        description="Scale to apply to `n_embd` to compute the dimensionality of inner layer of feedforward network",
        default=4,
    )

    @property
    def head_size(self):
        """
        Size of the attention head. This controls the size of the embedding space used
        for keys, queries, and values.

        During attention, we project our input tensors to a dimension of this size. For
        multi-attention, we concatenate all of the single-attention heads together,
        which means the final output is equal to `n_embd`.
        """
        return self.n_embd // self.n_head


class AttentionHead(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.config = config
        self.keys = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.queries = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.values = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x: torch.Tensor):
        # x = (B, T, C), where C = config.n_embd
        k: torch.Tensor = self.keys(x)  # B, T, head_size
        q: torch.Tensor = self.queries(x)  # B, T, head_size
        wei = q @ k.transpose(-2, -1) * self.config.head_size**-0.5  # B, T, T

        # Prevent positions from attending to subsequent positions.
        wei = wei.masked_fill(self.tril == 0, float('-inf'))

        # Output
        wei = F.softmax(wei, dim=-1)
        v: torch.Tensor = self.values(x)  # B, T, head_size
        out = wei @ v  # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out


class MultiAttentionHead(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.config = config
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config) for _ in range(config.n_head)]
        )
        self.linear_head = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x: torch.Tensor):
        h = torch.cat(
            [head(x) for head in self.attention_heads],
            dim=1
        )
        out = self.linear_head(h)
        return out


class Feedforward(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * config.ffw_inner_scale),
            nn.ReLU(),
            nn.Linear(config.n_embd * config.ffw_inner_scale, config.n_embd)
        )


class TransformerBlock(nn.Module):
    
    def __init__(self, config: TinyGptConfig):
        super().__init__()

        self.multi_attention_head = MultiAttentionHead(config)
        self.feedforward_nwk = Feedforward(config)

        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor):
        x = x + self.multi_attention_head(self.ln1(x))
        x = x + self.feedforward_nwk(self.ln2(x))
        return x


class TinyGpt(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln = nn.LayerNorm(config.n_embd)
        self.linear_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # x = (B, T)
        B, T = x.shape

        # Token + positional embeddings
        token_emb = self.token_embeddings(x)  # (B, T, C)
        pos_emb = self.pos_embeddings(torch.arange(T))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)

        # Compute logits
        x = self.blocks(x)  # (B, T, C)
        logits = self.linear_head(self.ln(x))

        # Compute loss
        if y is not None:
            B, T, C = logits.shape

            # Flatten the logits. Each row is the last character before the target.
            # e.g., if the sequence is "I am learning GPT", then the flatten operation
            # results in a X, y pairing that looks like:
            #   I --> _
            #   _ --> a
            #   a --> m
            #   m --> _
            #   _ --> l
            #   l --> e
            # ...
            #
            # This is intentional; with attention, the last character in our sequence
            # should have absorbed all the necessary context from previous characters.

            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
    
        return logits, loss
