import logging
import math
import os
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, DistributedSampler
from typing import Iterator

from ml.gpt.utils import (
    load_data,
    define_alphabet,
    build_dataset,
    START_TOKEN,
)


logger = logging.getLogger(__name__)


# Constants
WKDIR = Path(__file__).parent
GPT = WKDIR


def define_device(local_rank: int | None = None) -> str:
    if torch.cuda.is_available():
        if local_rank is not None:
            return f"cuda:{local_rank}"
        else:
            return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_ddp() -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise Exception("DDP can only run on CUDA!")

    # Initialize the process group using NCCL backend. Can potentially use the code
    # below to automatically select backend?
    #     acc = torch.accelerator.current_accelerator()
    #     backend = torch.distributed.get_default_backend_for_device(acc)
    init_process_group(backend="nccl")
    
    # Rank and world size are automatically calculated via `torchrun` and injected via
    # environment variables.
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup() -> None:
    destroy_process_group()


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
    max_lr: float = Field(
        description="Max learning rate",
        default=3e-4
    )
    lr_warmup_steps: int = Field(
        description="Number of steps used for learning rate's linear warmup.",
        default=5e3
    )
    lr_max_steps: int = Field(
        description="Max number of steps for scaling the learning rate (both warmup and cosine decay).",
        default=5e4,
    )
    weight_decay: float = Field(
        description="Weight decay for parameters that are >=2 dimensions (e.g., embeddings, params that participate in matrix multiplications, etc.)",
        default=0.1,
    )
    batch_size: int = Field(
        description="Batch size to use for training.",
        default=1024
    )
    accumulate_gradients: bool = Field(
        descrption="Whether to accumulate gradients over micro-batches",
        default=False,
    )
    micro_batch_size: int = Field(
        description="Micro batch size to use for training, used for gradient accumulation.",
        default=32,
    )
    num_epochs: int = Field(
        description="Number of epochs (complete passes through the dataset) to use for training.",
        default=10,
    )
    dropout: float = Field(
        description="Percent of neurons to randomly turn off during training.",
        default=0.1
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
    
    def get_accumulation_steps(self, world_size: int) -> int:
        if self.accumulate_gradients and self.batch_size % (self.micro_batch_size * world_size) > 0:
            raise Exception(f"`batch_size` must be divisible by (micro_batch_size * world_size). Got {self.batch_size % (self.micro_batch_size * world_size):.4f}.")

        if not self.accumulate_gradients:
            return 1
        return int(self.batch_size / (self.micro_batch_size * world_size))


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

        # Dropout
        self.dropout = nn.Dropout(p=config.dropout)

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

        # Attention
        # wei = q @ k.transpose(-2, -1) * self.config.head_size**-0.5  # B, nh, T, T
        # wei = wei.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf')) # B, nh, T, T
        # wei = F.softmax(wei, dim=-1)
        # out = wei @ v  # (B, nh, T, T) @ (B, nh, T, head_size) --> (B, nh, T, head_size)

        # Flash attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Confirm output shape
        expected_shape = torch.Size([B, self.config.n_head, T, self.config.head_size])
        if out.shape != expected_shape:
            raise Exception(f"Unexpected shape of softmax weights @ values {out.shape}, expected: {expected_shape}.")

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.dropout(out)
        return out


class MLP(nn.Module):

    def __init__(self, config: TinyGptConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.ffw_inner_scale)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(config.n_embd * config.ffw_inner_scale, config.n_embd)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
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

    def __init__(self, config: TinyGptConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device

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
    
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Dimension can be 1 if computing validation loss or performing inference
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (B, T), where B = 1

        # x = (B, T)
        B, T = x.shape

        # Token + positional embeddings
        token_emb = self.wte(x)  # (B, T, C)
        pos_emb = self.wpe(torch.arange(T, device=self.device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)

        # Compute logits
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(self.ln(x))  # (B, T, num_tokens)

        # Compute loss
        if y is not None:
            B, T, num_tokens = logits.shape

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
            loss = F.cross_entropy(logits.view(B*T, num_tokens), y.view(B*T))
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
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)  # (1, T)

        for _ in range(max_tokens):
            # idx should be a 1-D tensor of indices
            idx_cond = idx[:, -self.config.block_size:]  # (1, T)

            # Compute logits. Focus on logits of last character.
            logits, _ = self(idx_cond)  # (1, T, num_tokens)
            logits = logits[:, -1, :]  # (1, num_tokens)
            if top_k:
                v, ix = torch.topk(logits, top_k).values  # (1,k) (1,k)
                probs = F.softmax(v, dim=-1)  # (1,k)
                k_choice = torch.multinomial(probs, num_samples=1)  # (1,1)
                next_idx = ix.gather(-1, k_choice)
            else:
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)  # (1,1)

            print(self.decode(next_idx.item()), end="")
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def configure_optimizers_with_initial_lr(parameters: Iterator[Parameter], lr: float | torch.Tensor, weight_decay: float,) -> torch.optim.AdamW:
    # Parameters that are at least 2-dimensions will be weight-decayed. 1-D
    # tensors (e.g., biases, layer norms) are not weight-decayed.
    params_grad: list[nn.Parameter] = [p for p in parameters if p.requires_grad]
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


def get_lr(step: int, config: TinyGptConfig) -> float:
    """
    Learning rate with:
        1) Linear warmup
        2) Cosine decay
    """
    min_lr = config.max_lr * 0.1
    decay_steps = config.lr_max_steps - config.lr_warmup_steps

    # Linear increase until we hit warmup steps
    if step < config.lr_warmup_steps:
        return config.max_lr * (step+1) / config.lr_warmup_steps

    # If we have exceeded the max steps used for decay, use minimum LR
    elif step > config.lr_max_steps:
        return min_lr
    
    # Otherwise, cosine decay. T_cur = the current step number used in decay, T_max
    # = total steps used for decay. Note that T_cur is different than `step`, since
    # `step` also counts warmup steps.
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    else:
        coeff = 0.5 * (1.0 + math.cos(math.pi * (step - config.lr_warmup_steps) / decay_steps))
        return min_lr + coeff * (config.max_lr - min_lr)


if __name__ == "__main__":
    # Distributed data processing
    local_rank, global_rank, world_size = setup_ddp()
    is_master_process = global_rank == 0

    # Use TensorFloat32 datatype (1 sign bit, 8 exponent bits, 10 mantissa bits). This
    # is lower precision that FP32 (23 mantissa bits), which results in faster training
    # by less precision. In practice, the loss is precision is not noticeable.
    torch.set_float32_matmul_precision('high')
    torch.backends.fp32_precision = "tf32"  # allows TF32 for all applicable ops globally

    # Raw data
    text_data = load_data(WKDIR, GPT)
    token_set = define_alphabet(text_data)
    
    # Hyperparameters
    model_config = TinyGptConfig(
        token_set=token_set,
        accumulate_gradients=True,
        batch_size=32,
        micro_batch_size=8
    )
    
    # Datasets
    datasets = build_dataset(text_data, model_config._chars_to_index_map, context_len=model_config.block_size)

    # Model
    device = define_device(local_rank)
    model = TinyGpt(model_config, device=device).to(device)
    ddp_model = DDP(model, device_ids=[global_rank])
    ddp_model.compile()
    
    total_params = sum([p.numel() for p in model.parameters()])
    if is_master_process:
        print(f"Number of parameters: {total_params}")

    # Optimizer
    optimizer = configure_optimizers_with_initial_lr(
        ddp_model.parameters(),
        model_config.max_lr,
        model_config.weight_decay
    )

    # Training data
    sampler = DistributedSampler(
        dataset=datasets["train"],
        num_replicas=world_size,
        rank=global_rank
    )
    training_data_loader = DataLoader(
        dataset=datasets["train"],
        batch_size=(
            model_config.micro_batch_size
            if model_config.accumulate_gradients
            else model_config.batch_size
        ),
        sampler=sampler,
        shuffle=False,
    )
    val_x, val_y = datasets["val"][0]
    val_x, val_y = val_x.to(device), val_y.to(device)

    # Training loop
    running_loss = 0.0
    training_loss = []
    validation_loss = []
    for epoch_num in range(model_config.num_epochs):
        epoch_loss = 0.0

        start_time = time.time()

        ddp_model.train()
        for i, (x, y) in enumerate(training_data_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Accumulate gradients. Note that if we are not accumulating gradients, then
            # accumulation_steps will just be 1.
            loss_accum = 0.0
            accum_steps = model_config.get_accumulation_steps(world_size)
            for j in range(accum_steps):
                # Forward pass
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = ddp_model(x, y)

                # We want to make sure our loss reporesents average across full batch,
                # not each micro-batch.
                loss = loss / accum_steps
                loss_accum += loss

                # Backward pass. DDP is responsible for syncing gradients during the
                # backward pass (i.e., collecting gradients from each of the ranks,
                # averaging them, and depositing those averages onto the ranks). We
                # don't want to do that for each micro-step, since this is wasteful.
                if j < accum_steps - 1:
                    with ddp_model.no_sync():
                        loss.backward()
                else:
                    # Perform AllReduce operation once we have completed all the
                    # micro-steps.
                    loss.backward()

            # At this point, loss_accum is the accumulated loss from just a single rank.
            # Average it across all ranks (i.e., exactly what DDP does for the grads
            # with loss.backward()).
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            # LR
            lr = get_lr(i, model_config)
            for param_groups in optimizer.param_groups:
                param_groups["lr"] = lr

            # Clip gradients
            norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)

            # Step and synchronize
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            # Batch loss statistics
            running_loss += loss_accum.detach()
            epoch_loss += loss_accum.detach()

            # Every 100 batches on the master process, log some statistics
            if is_master_process and i > 0 and (i+1) % 100 == 0:
                # Number of tokens processed / second
                end_time = time.time()
                batch_elapsed = end_time - start_time
                total_tokens = (
                    100  # 100 batches
                    * model_config.batch_size  # Each batch has `batch_size` examples
                    * model_config.block_size  # Each example in batch has block_size tokens
                    * world_size
                )
                total_tokens_per_sec = f"Tokens/sec: {(total_tokens / batch_elapsed):.4f}"
                
                # Total / average step time
                total_step_time = f"Batch time: {(end_time - start_time)*1000:.4f}ms"
                avg_step_time = f"Avg Step: {(end_time - start_time)*1000/100:.4f}ms"
                start_time = time.time()

                # Average loss
                average_loss = f"Average Loss: {(running_loss/100).item():.4f}"

                # Gradient norm
                gradient_norm = f"Gradient Norm: {norm:.4f}"
                if is_master_process:
                    print(f"Epoch {epoch_num+1:02d} | Batch {(i+1-100):04d}-{i:04d} | {average_loss} | {gradient_norm} | {total_tokens_per_sec} | {total_step_time} | {avg_step_time}")

                running_loss = 0
        
        # Epoch loss
        training_loss.append(epoch_loss / len(training_data_loader))

        # Validation loss
        ddp_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                val_logits, val_loss = ddp_model(val_x, val_y)
        validation_loss.append(val_loss.item())

    # model.generate(torch.tensor([model.encode(START_TOKEN)]))
