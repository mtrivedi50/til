from pydantic import BaseModel, Field
import torch
from torch import nn
import torch.nn.functional as F
import math


class TinyGptConfig(BaseModel):
    vocab_size: int = Field(
        description="Total size of vocab.",
        default=27,
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
        default=4,
    )
    n_layer: int = Field(
        description=" ".join([
            "The number of layers to use within the decoder.",
            "An individual layer is a masked multi-attention head followed by a feed-forward MLP."
        ]),
        ge=1,
        default=4,
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
        head_size = self.n_embd // self.n_head
        if self.n_embd % self.n_head != 0:
            raise Exception("`n_embd` must be divisible by `n_head`!")
        return head_size


class CasualSelfAttention(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.config = config

        # Project n_embd to three vectors. We will split these and then make the number
        # of attention heads a batch dimension.
        self.kqv = nn.Linear(config.n_embd, config.n_embd*3, bias=False)

        # Output projection, per multi-head attention definition in "Attention is all
        # you need" paper.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Lower-triangular matrix for masking. Note viewing by (1, 1, block_size,
        # block_size) works because we broadcast into the 1 dimensions.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor):
        # x = (B, T, C), where C = config.n_embd
        B, T, C = x.shape
        k, q, v = self.kqv(x).split(C, dim=-1)  # 3 tensors, each of B, T, C
        k: torch.Tensor = k.view(B, T, self.config.n_head, self.config.head_size).transpose(-2, -1)  # B, nh, T, head_size
        q: torch.Tensor = q.view(B, T, self.config.n_head, self.config.head_size).transpose(-2, -1)  # B, nh, T, head_size
        v: torch.Tensor = v.view(B, T, self.config.n_head, self.config.head_size).transpose(-2, -1)  # B, nh, T, head_size
        
        wei = q @ k.transpose(-2, -1) * self.config.head_size**-0.5  # B, nh, T, T

        # Prevent positions from attending to subsequent positions.
        wei = wei.masked_fill(self.tril[:,:,T,T] == 0, float('-inf')) # B, nh, T, T

        # Softmax and output
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, nh, T, T) @ (B, nh, T, head_size) --> (B, nh, T, head_size)
        out = out.transpose(-2, -1).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.ffw_inner_scale)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(config.n_embd * config.ffw_inner_scale, config.n_embd)

    def forward(self, x: torch.Tensor):
        return self.network(x)


class TransformerBlock(nn.Module):
    
    def __init__(self, config: TinyGptConfig):
        super().__init__()

        self.c_attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor):
        # Skip connections (aka residual connections).
        x = x + self.c_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGpt(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Weight sharing. Note that lm_head.weight.shape = (vocab_size, n_embd), which
        # matches exactly the shape of wte.weight.
        self.lm_head.weight = self.wte.weight
        if self.lm_head.weight.data_ptr() != self.wte.weight.data_ptr():
            raise Exception("Not properly sharing weights between input embedding layer and linear head!")
        
        self.apply(self._init_weights)

        # Scale weight of residual layers by a factor of 1/sqrt(n), where n is the
        # number of residual conections. The transformer block has two residual
        # connections and there are n_layer transformer blocks in our architecture.

        # GPT2 scales the weights of the linear layers that feed into the residual
        # connections, so we do that here.
        for param_name, param in self.named_parameters():
            if param_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        """
        GPT2 initialization
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # x = (B, T)
        B, T = x.shape

        # Token + positional embeddings
        token_emb = self.wte(x)  # (B, T, C)
        pos_emb = self.wpe(torch.arange(T))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)

        # Compute logits
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(self.ln(x))

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
            # should have absorbed all the necessary context by "talking to" previous
            # characters.
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
    
        return logits, loss


if __name__ == "__main__":
    # Training loop...
    pass