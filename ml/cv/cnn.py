"""
Basic CNN for action recognition.
"""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from torchvision.datasets import UCF101
import math
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal
from torch.utils.data import DataLoader, DistributedSampler, Dataset, random_split
from torch.utils.data.dataset import Subset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import os
import time
import random


# Paths / constants
WKDIR = Path(__file__).parent
DATA = WKDIR / 'data'


# ===== Distributed training setup ===== #

def setup_ddp() -> tuple[int | None, int | None, int | None]:
   if torch.cuda.is_available():
      init_process_group(backend="nccl")

      # Rank and world size are automatically calculated via `torchrun` and injected via
      # environment variables.
      #
      #    Machine 1 (Node 0)          Machine 2 (Node 1)
      # ┌─────────────────────┐     ┌─────────────────────┐
      # │  GPU 0  │  GPU 1    │     │  GPU 0  │  GPU 1    │
      # │ rank=0  │ rank=1    │     │ rank=2  │ rank=3    │
      # │ local=0 │ local=1   │     │ local=0 │ local=1   │
      # └─────────────────────┘     └─────────────────────┘
      # world_size = 4
      local_rank = int(os.environ.get("LOCAL_RANK", 0))  # unique ID for GPUs within current node
      global_rank = int(os.environ.get("RANK", 0))  # unique ID for GPUs across all nodes
      world_size = int(os.environ.get("WORLD_SIZE", 1))  # total number of GPUs across all nodes

      torch.cuda.set_device(local_rank)

      return local_rank, global_rank, world_size
   
   return None, 0, None  # so current machine is automatically master process


def cleanup():
   destroy_process_group()


def define_device(local_rank: int | None = None):
   if torch.cuda.is_available():
      if local_rank is not None:
         return f"cuda:{local_rank}"
      else:
         return "cuda"
   elif torch.mps.is_available():
      return "mps"
   else:
      return "cpu"


# ===== Load data ===== #

def load_ucf101(
   root: Path,
   annotation_path: Path,
   frames_per_clip: int = 8,
   steps_between_clips: int = 1,
   output_format: str = "TCHW",
   train_size: float = 0.98
) -> tuple[Subset, Subset, Subset]:
   ucf101 = UCF101(
      root=root,
      annotation_path=annotation_path,
      frames_per_clip=frames_per_clip,
      step_between_clips=steps_between_clips,
      output_format=output_format
   )
   print("==================== DATA SUMMARY ====================")
   print(f"There are {len(ucf101)} video clips. Each clip is of type {ucf101[0].__class__.__name__}.")
   print(f"Each {ucf101[0].__class__.__name__} has length {len(ucf101[0])}, which corresponds to video, audio, label.")
   print("Example clip:")
   print(f"Video shape: {ucf101[0][0].shape}")
   print(f"Audio shape: {ucf101[0][1].shape}")
   print(f"Label: {ucf101[0][2]}")
   print("======================================================")

   # Training / validation / test split
   test_val_size = 1 - train_size
   test_size, val_size = test_val_size / 2, test_val_size / 2
   train_dataset, test_dataset, val_dataset = random_split(ucf101, [train_size, test_size, val_size])
   return train_dataset, test_dataset, val_dataset


class UCF101Dataset(Dataset):

   def __init__(self, device: str, ucf101: UCF101):
      self.device = device
      self.ucf101 = ucf101

   def __len__(self):
        return len(self.ucf101)
   
   def _get(self, idx):
      video, _, label = self.ucf101[idx]
      # Conv3d expects: (C, D, H, W). We load (D, C, H, W) → permute to (batch,
      # channels, frames, height, width)
      video = video.transpose(0, 1)
      return video.float(), label
   
   def __getitem__(self, idx):
      try:
         return self._get(idx)

      except RuntimeError as e:
         # Corrupted video? Seems sporadic. Should investigate further but for now
         # choose a random index.
         # https://github.com/meta-pytorch/torchcodec/issues
         if "Requested next frame while there are no more frames left to decode" in str(e):
            # Keep track
            with open(WKDIR / f"corrupted_{self.device}.txt", "a") as f:
               f.write(idx)
               
            idx = random.randint(0, len(self.ucf101)-1)
            return self._get(idx)
         raise


# ===== Modeling ===== #

class ModelParams(BaseModel):
   kernel_size: tuple[int, int, int]
   padding: tuple[int, int, int] = Field(default=(1,1,1))
   stride: tuple[int, int, int] = Field(default=(1,1,1))

   def create_module(self) -> nn.Module:
      pass

   def compute_output_dim(self, input_dims: tuple[int, int, int]) -> tuple[int, int, int]:
      output = []
      for f, d, s, p in zip(self.kernel_size, input_dims, self.stride, self.padding):
         output.append(
            math.floor((d + 2*p - f) / s + 1)
         )
      return output[0], output[1], output[2]


class Conv3dParams(ModelParams):
   in_channels: int
   out_channels: int
   kernel_size: tuple[int, int, int] = Field(default=(3,3,3))

   @model_validator(mode="before")
   @classmethod
   def _convert_ints_to_tuple(cls, data: dict[str, Any]) -> dict[str, Any]:
      for key in ["kernel_size", "padding", "stride"]:
         if key in data and isinstance(data[key], int):
            val = data[key]
            data[key] = (val, val, val)
      return data

   def create_module(self) -> nn.Module:
      return nn.Conv3d(
         in_channels=self.in_channels,
         out_channels=self.out_channels,
         kernel_size=self.kernel_size,
         stride=self.stride,
         padding=self.padding,
      )


class MaxPool3dParams(ModelParams):
   kernel_size: tuple[int, int, int] = Field(default=(2,2,2))
   stride: tuple[int, int, int] = Field(default=(2,2,2))
   padding: tuple[int, int, int] = Field(default=(0,0,0))

   @model_validator(mode="before")
   @classmethod
   def _convert_ints_to_tuple(cls, data: dict[str, Any]) -> dict[str, Any]:
      for key in ["kernel_size", "padding", "stride"]:
         if key in data and isinstance(data[key], int):
            val = data[key]
            data[key] = (val, val, val)
      return data

   def create_module(self) -> nn.Module:
      return nn.MaxPool3d(
         kernel_size=self.kernel_size,
         stride=self.stride,
         padding=self.padding,
      )


class ModelBlock(BaseModel):
   num_convs: int
   conv3d_params: Conv3dParams
   nonlin: Literal["relu", "gelu"] | None
   maxpool3d_params: MaxPool3dParams

   def create_module(self) -> nn.Module:
      layers = []
      for i in range(self.num_convs):
         # For sequential convolutional layers, only the last layer should have
         # `out_channels` output channels. All other ones should have `out_channels` =
         # `in_channels.``
         if i < self.num_convs - 1:
            out_channels = self.conv3d_params.in_channels
         else:
            out_channels = self.conv3d_params.out_channels
         conv3d_params_copy = self.conv3d_params.model_copy(
            update={"out_channels": out_channels}
         )
         layers.append(conv3d_params_copy.create_module())
         
         if self.nonlin == "relu":
            layers.append(nn.ReLU())
         elif self.nonlin == "gelu":
            layers.append(nn.GELU())
      layers.append(self.maxpool3d_params.create_module())
      return nn.Sequential(*layers)
   
   def compute_output_dim(self, input_dims: tuple[int, int, int]) -> tuple[int, int, int]:
      # Calculate output dims as we apply sequential convolutions. Note that, for
      # filters of size 3x3x3 with padding = 1 and stride = 1, the dimensions do not
      # change.
      cur = input_dims
      for _ in range(self.num_convs):
         cur = self.conv3d_params.compute_output_dim(cur)
      return self.maxpool3d_params.compute_output_dim(cur)


@dataclass
class ValidationStats:
   loss: float
   accuracy: float


class ActionRecognitionModel(nn.Module):

   def __init__(
      self,
      input_dims: tuple[int, int, int],
      num_classes: int,
      blocks: list[ModelBlock],
   ):
      super().__init__()

      self._num_blocks = len(blocks)

      output_dims = input_dims
      for i, b in enumerate(blocks):
         setattr(self, f"block{i}", b.create_module())
         output_dims = b.compute_output_dim(output_dims)

      # FC
      self.flatten = nn.Flatten()
      self.fc1 = nn.Sequential(
         nn.Linear(
            math.prod(output_dims) * blocks[-1].conv3d_params.out_channels,
            256
         ),
         nn.BatchNorm1d(num_features=256),
      )
      self.fc2 = nn.Sequential(
         nn.Linear(
            256,
            32,
         ),
         nn.BatchNorm1d(num_features=32),
      )
      self.fc3 = nn.Linear(32, num_classes)

      self._input_shape = (blocks[0].conv3d_params.in_channels, *input_dims)

   def print_sequential_arch(self, layer_num: int, layer: nn.Sequential, x: torch.Tensor):
      for i, submodule in enumerate(layer):
         in_shape = tuple(x.shape[1:])
         x = submodule(x)
         out_shape = tuple(x.shape[1:])
         print(f"├── {layer_num}.{i}: ({submodule.__class__.__name__}): {in_shape} -> {out_shape}")

   def print_arch(self, device: torch.device):
      def _print_layer(x: torch.Tensor, layer: nn.Module):
         in_shape = tuple(x.shape[1:])  # ignore batch size
         out = layer(x)
         out_shape = tuple(out.shape[1:])
         print(f"({layer.__class__.__name__}): {in_shape} -> {out_shape}")
         return out

      x = torch.zeros(2, *self._input_shape)  # use 2 b/c a batch of 1 breaks BatchNorm
      x = x.to(device)
      with torch.no_grad():
         for i in range(self._num_blocks):
            layer = getattr(self, f"block{i}")
            out = _print_layer(x, layer)

            # Print architecture within the Sequential module
            if isinstance(layer, nn.Sequential):
               self.print_sequential_arch(i, layer, x)

            x = out
         
         # Flatten
         x = _print_layer(x, self.flatten)

         # FCs
         x = _print_layer(x, self.fc1)
         x = _print_layer(x, self.fc2)
         _print_layer(x, self.fc3)

   def forward(self, x: torch.Tensor):
      out = x
      
      # Convolution - MaxPool blocks
      for i in range(self._num_blocks):
         out = getattr(self, f"block{i}")(out)
      
      out = self.flatten(out)
      
      # FCs
      out = self.fc1(out)
      out = self.fc2(out)
      return self.fc3(out)
   

def create_activations_forward_hooks(
   model: nn.Module
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
   activations = {}

   def _make_forward_hook(name: str):
      def _hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
         # Convolutional layer - since we apply ReLU, only look at output > 0
         if output.ndim == 5:
            alive = (output > 0).float().mean(dim=[0, 2, 3, 4])  # shape: (channels,)
            magnitude = output.abs().mean(dim=[0, 2, 3, 4])  # shape: (channels,)
         
         # FC layers - we apply batch norm, so the fraction of alive neurons should
         # hover around 50%.
         else:
            alive = (output > 0).float().mean(dim=0)
            magnitude = output.abs().mean(dim=0)

         activations[name] = {"alive": alive, "magnitude": magnitude}

      return _hook

   for m in model.named_modules():
      if m[0]:
         m[1].register_forward_hook(_make_forward_hook(m[0]))

   return (model, activations)


def evaluate(
   ddp_model: DDP,
   val_dataloader: DataLoader,
   batch_size: int,
   device: str,
) -> ValidationStats:
   running_loss = 0.0
   running_correct = 0
   with torch.no_grad():
      for batch in val_dataloader:
         videos, labels = batch[0], batch[1]
         videos, labels = videos.to(device), labels.to(device)
         preds = ddp_model(videos)  # N x num_classes

         # Loss
         loss = nn.CrossEntropyLoss()
         val_loss = loss(preds, labels).item()  # default is mean across the batch
         running_loss += (val_loss * batch_size)

         # Accuracy
         probs = F.softmax(preds, dim=1)  # probability over number of classes
         predicted_labels = probs.argmax(dim=1)
         correct = (predicted_labels == labels).to(torch.int).sum().item()
         running_correct += correct

   return ValidationStats(
      loss=1.0 * running_loss / (len(val_dataloader) * batch_size),
      accuracy=1.0 * running_correct / (len(val_dataloader) * batch_size)
   )


def train(
   device: torch.device,
   train_dataset: Subset,
   val_dataset: Subset,
   world_size: int,
   global_rank: int,
   ddp_model: DDP,
   activations: dict[str, torch.Tensor],
   batch_size: int = 32,
   num_epochs: int = 10_000,
   lr: float = 1e-3,
   weight_decay: float = 1e-4,
):
   sampler = DistributedSampler(
      dataset=train_dataset,
      num_replicas=world_size,
      rank=global_rank,
      shuffle=True,
      drop_last=True
   )
   dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      sampler=sampler,
      shuffle=False,  # shuffle handled by sampler
      drop_last=True,
   )
   val_sampler = DistributedSampler(
      dataset=val_dataset,
      num_replicas=world_size,
      rank=global_rank,
      shuffle=False,
      drop_last=True,
   )
   val_dataloader = DataLoader(
      dataset=val_dataset,
      batch_size=batch_size,
      sampler=val_sampler,
      shuffle=False,
      drop_last=True,
   )

   writer = SummaryWriter()

   # AdamW optimizer. Use constant LR for now, and apply weight decay to all params,
   # including batch norm running mean / std. dev + biases
   optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=weight_decay)

   global_step = 0
   
   criterion = nn.CrossEntropyLoss()

   for epoch_num in range(num_epochs):
      # To ensure random shuffling at every epoch
      sampler.set_epoch(epoch_num)

      # Initial evaluation
      # We expect initial loss to be ~1% (guessing randomly from 100 classes)
      ddp_model.eval()
      val_stats = evaluate(
         ddp_model,
         val_dataloader,
         batch_size,
         device,
      )
      ddp_model.train()
      print(f"val loss: {val_stats.loss:.4f} | val accuracy: {val_stats.accuracy:.4f}")
      
      # Training mode      
      ddp_model.train()

      training_step = 0
      running_loss = 0.0
      epoch_loss = 0.0
      start_time = time.time()

      for batch in dataloader:
         optimizer.zero_grad(set_to_none=True)

         # Conv3d expects: (N, C, D, H, W). We load (N, D, C, H, W) → permute to (batch,
         # channels, frames, height, width)
         videos, labels = batch[0], batch[1]
         videos, labels = videos.to(device), labels.to(device)

         # Forward pass
         out = ddp_model(videos)
         loss = criterion(out, labels)

         # Loss statistics
         running_loss += loss
         epoch_loss += loss

         # Backwards pass. DDP is responsible for syncing gradients during the backwards
         # pass (collecting gradients from each of the ranks, averaging them, depositing
         # averages onto the ranks).
         loss.backward()

         # Training statistics
         global_step += 1
         training_step += 1
         if is_master_process and training_step % 100 == 0:
            end_time = time.time()
            elapsed = end_time - start_time

            # Activations
            for name, activation_stats in activations.items():
               alive = activation_stats["alive"]
               magnitude = activation_stats["magnitude"]
               writer.add_histogram(f"activations/{name}/alive", alive, global_step=global_step)
               writer.add_histogram(f"activations/{name}/magnitude", magnitude, global_step=global_step)
            
            # Gradients
            # Total gradient norm
            total_norm = torch.nn.utils.get_total_norm(model.parameters())
            for name, module in model.named_modules():
               parameters = [p for p in module.parameters(recurse=False) if p.grad is not None]
               if parameters:
                  layer_grad_norm = torch.stack([p.grad.norm() ** 2 for p in parameters]).sum().sqrt()
                  writer.add_scalar(f"gradients/{name}/layer_grad_norm", layer_grad_norm, global_step=global_step)
         
            if training_step % 1000 == 0:
               ddp_model.eval()
               val_stats = evaluate(
                  ddp_model,
                  val_dataloader,
                  batch_size,
                  device,
               )
               ddp_model.train()
            else:
               # Dummy stats for now
               val_stats = ValidationStats(
                  loss=float("inf"),
                  accuracy=float("inf")
               )

            msg = [
               f"epoch: {epoch_num:5d}",
               f"global step: {global_step:5d}",
               f"training_step: {training_step:5d}",
               f"elapsed (n=100): {elapsed:.4f}",
               f"total norm: {total_norm:.4f}",
               f"avg loss (n=100): {(running_loss/100.0):.4f}",
               f"val loss: {val_stats.loss:.4f}",
               f"val accuracy: {val_stats.accuracy:.4f}"
            ]
            print(" | ".join(msg))
            running_loss = 0.0
            start_time = time.time()

         optimizer.step()

      print(f"epoch {epoch_num:5d}| global step: {global_step:5d} | avg epoch loss: {1.0 * epoch_loss/training_step}")


if __name__ == "__main__":
   local_rank, global_rank, world_size = setup_ddp()
   device = define_device(local_rank)
   is_master_process = global_rank == 0

   # Model
   model = ActionRecognitionModel(
      input_dims=(8, 240, 320),
      num_classes=101,
      blocks=[
         ModelBlock(
            **{
               "num_convs": 3,
               "nonlin": "relu",
               "conv3d_params": {
                  "in_channels": 3,
                  "out_channels": 64
               },
               "maxpool3d_params": {}
            },
         ),
         ModelBlock(
            **{
               "num_convs": 3,
               "nonlin": "relu",
               "conv3d_params": {
                  "in_channels": 64,
                  "out_channels": 32
               },
               "maxpool3d_params": {}
            }
         ),
         ModelBlock(
            **{
               "num_convs": 3,
               "nonlin": "relu",
               "conv3d_params": {
                  "in_channels": 32,
                  "out_channels": 8,
                  "kernel_size": (2,5,5),
               },
               "maxpool3d_params": {}
            }
         ),
      ]
   )
   model, activations = create_activations_forward_hooks(model)
   model.to(device)
   model = torch.compile(model)
   ddp_model = DDP(model, device_ids=[local_rank])

   # Model parameters
   if is_master_process:
      print("================= MODEL ARCHITECTURE =================")
      model.print_arch(device)
      total_params = sum([p.numel() for p in model.parameters()])
      print(f"Total parameters: {total_params:,}")
      print("======================================================")

   train_subset, test_subset, val_subset = load_ucf101(
      root=DATA / 'ucf101',
      annotation_path=DATA / 'annotations'
   )
   train_dataset = UCF101Dataset(train_subset)
   val_dataset = UCF101Dataset(val_subset)
   if is_master_process:
      print("==================== DATA SUMMARY ====================")
      print(f"Training dataset: {len(train_subset)} videos")
      print(f"Validation dataset: {len(val_subset)} videos")
      print(f"Test dataset: {len(test_subset)} videos")
      print("======================================================")
   
   try:
      train(
         device=device,
         train_dataset=train_dataset,
         val_dataset=val_dataset,
         world_size=world_size,
         global_rank=global_rank,
         ddp_model=ddp_model,
         activations=activations,
      )
   except Exception:
      raise
   finally:
      destroy_process_group()
