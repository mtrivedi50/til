import logging
import math
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.gpt.utils import (
    load_data,
    define_alphabet,
    build_dataset,
    START_TOKEN,
)


logger = logging.getLogger(__name__)


WKDIR = Path(__file__).parent.parent.parent
GPT = WKDIR / "ml" / "gpt"


class TinyGptConfig(BaseModel):
    token_set: list[str] = Field(
        description="Set of all possible tokens in our corpus.",
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
    # Training hyperparameters
    lr: float = Field(
        description="Default learning rate",
        default=1e-3
    )
    weight_decay: float = Field(
        description="Weight decay for parameters that are >=2 dimensions (e.g., embeddings, params that participate in matrix multiplications, etc.)",
        default=0.1,
    )
    batch_size: int = Field(
        description="Batch size to use for training.",
        default=32,
    )
    num_epochs: int = Field(
        description="Number of epochs (complete passes through the dataset) to use for training.",
        default=10,
    )

    _chars_to_index_map: dict[str, int] = PrivateAttr()
    _index_to_chars_map: dict[int, str] = PrivateAttr()

    @model_validator(mode="after")
    def create_private_attrs(self) -> "TinyGptConfig":
        self._chars_to_index_map = {}
        self._index_to_chars_map = {}
        for i, c in enumerate(self.token_set):
            self._chars_to_index_map[c] = i
            self._index_to_chars_map[i] = c
        return self

    @property
    def num_tokens(self):
        return len(self.token_set)

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
        k: torch.Tensor = k.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)  # B, nh, T, head_size
        q: torch.Tensor = q.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)  # B, nh, T, head_size
        v: torch.Tensor = v.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)  # B, nh, T, head_size

        wei = q @ k.transpose(-2, -1) * self.config.head_size**-0.5  # B, nh, T, T

        # Prevent positions from attending to subsequent positions.
        wei = wei.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf')) # B, nh, T, T

        # Softmax and output
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, nh, T, T) @ (B, nh, T, head_size) --> (B, nh, T, head_size)

        # Confirm output shape
        expected_shape = torch.Size([B, self.config.n_head, T, self.config.head_size])
        if out.shape != expected_shape:
            raise Exception(f"Unexpected shape of softmax weights @ values {out.shape}, expected: {expected_shape}.")

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.ffw_inner_scale)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(config.n_embd * config.ffw_inner_scale, config.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x


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
        self.config = config

        self.wte = nn.Embedding(config.num_tokens, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.num_tokens)

        # Weight sharing. Note that lm_head.weight.shape = (num_tokens, n_embd), which
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

    def configure_optimizers(self, lr: float | torch.Tensor, weight_decay: float,) -> torch.optim.AdamW:
        # Parameters that are at least 2-dimensions will be weight-decayed. 1-D
        # tensors (e.g., biases, layer norms) are not weight-decayed.
        params_grad: list[nn.Parameter] = [p for p in self.parameters() if p.requires_grad]
        p_decay = []
        p_no_decay = []
        for p in params_grad:
            if p.dim() >= 2:
                p_decay.append(p)
            else:
                p_no_decay.append(p)
        
        param_groups = [
            {
                "weight_decay": weight_decay,
                "params": p_decay,
            },
            {
                "weight_decay": 0.0,
                "params": p_no_decay,
            },
        ]
        optim = torch.optim.AdamW(params=param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        return optim

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x = (B, T)
        B, T = x.shape

        # Token + positional embeddings
        token_emb = self.wte(x)  # (B, T, C)
        pos_emb = self.wpe(torch.arange(T))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)

        # Compute logits
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(self.ln(x))  # (B, T, num_tokens)

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

    def encode(self, char: str) -> int:
        return self.config._chars_to_index_map[char]

    def decode(self, idx: int) -> str:
        return self.config._index_to_chars_map[idx] 

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_tokens: int = 1000, top_k: int | None = None) -> torch.Tensor:
        """
        Sample from the model to generate text.
        """
        for _ in range(max_tokens):
            # idx should be a 1-D tensor of indices
            idx = idx if idx.shape[0] <= self.config.block_size else idx[-self.config.block_size:]

            # reshape to be 2-D, since our model expects batches
            idx = idx.view(1, idx.shape[0])

            # Compute logits. Focus on logits of last character.
            logits, _ = self(idx)  # (T, num_tokens)
            logits = logits[-1, :]  # num_tokens
            if top_k:
                logits = torch.topk(logits, top_k).values

            # Sample from probability distribution
            probs = F.softmax(logits)
            next_idx = torch.multinomial(probs, num_samples=1)[0].item()
            print(self.decode(next_idx), end="")
            idx = torch.cat([idx.flatten(), torch.tensor([next_idx])])
        return idx


if __name__ == "__main__":
    # Raw data
    text_data = load_data(WKDIR, GPT)
    token_set = define_alphabet(text_data)
    
    # Hyperparameters
    model_config = TinyGptConfig(token_set=token_set)
    
    # Datasets
    datasets = build_dataset(text_data, model_config._chars_to_index_map, context_len=model_config.block_size)

    # Model
    model = TinyGpt(model_config)
    optimizer = model.configure_optimizers(model_config.lr, model_config.weight_decay)
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {total_params}")

    # Training data
    training_data_loader = DataLoader(
        dataset=datasets["train"],
        batch_size=model_config.batch_size,
        shuffle=True,
    )
    val_x, val_y = datasets["val"][0]

    # Training loop
    running_loss = 0.0
    training_loss = []
    validation_loss = []
    for epoch_num in range(model_config.num_epochs):
        epoch_loss = 0.0

        model.train()
        for i, (x, y) in enumerate(training_data_loader):
            optimizer.zero_grad()

            # Forward pass
            logits, loss = model(x, y)

            # Backward pass
            loss.backward()

            # Batch loss statistics
            running_loss += loss.item()
            if i > 0 and (i+1) % 100 == 0:
                # Average loss
                average_loss = f"Average Loss: {running_loss/100:.4f}"

                # Gradient norm
                l2_norm = 0.0
                grads = [param.grad.detach().flatten() for param in model.parameters()]
                l2_norm = torch.cat(grads).norm(2)
                gradient_norm = f"Gradient Norm: {(l2_norm ** 0.5):.4f}"
                print(f"Batch {(i+1-100):03d}-{i:03d} | {average_loss} | {gradient_norm}")

                epoch_loss += running_loss
                running_loss = 0
            
            optimizer.step()
        
        # Epoch loss
        training_loss.append(epoch_loss / len(training_data_loader))

        # Validation loss
        val_logits, val_loss = model(val_x.view(1,-1), val_y)
        validation_loss.append(val_loss.item())

    model.generate(torch.tensor([model.encode(START_TOKEN)]))
