"""
Basic CNN for action recognition.
"""

import torch
from torch import nn
from pathlib import Path
from torchvision.datasets import UCF101
import math
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal
from torch.utils.data import IterableDataset, DataLoader


# Paths / constants
WKDIR = Path(__file__).parent
DATA = WKDIR / 'data'


# ===== Load data ===== #

def load_ucf101(
   root: Path,
   annotation_path: Path,
   frames_per_clip: int = 8,
   steps_between_clips: int = 1,
   output_format: str = "TCHW"
):
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
   return ucf101


class UCF101Dataset(IterableDataset):

   def __init__(self, ucf101: UCF101):
      self.ucf101 = ucf101

   def __iter__(self):
      indices = torch.randperm(len(self.ucf101)).to_list()
      for i in indices:
         item = ucf101[i]

         # Conv3d expects: (N, C, D, H, W). For a single video, we load (D, C, H, W) →
         # permute to (batch, channels, frames, height, width)
         video = item[0].transpose(0, 1).to(torch.float32)
         label = item[2]
         yield video, label


def create_dataset(
   root: Path,
   annotation_path: Path,
   frames_per_clip: int = 8,
   steps_between_clips: int = 1,
   output_format: str = "TCHW"
) -> IterableDataset:
   ucf101 = load_ucf101(
      root,
      annotation_path,
      frames_per_clip,
      steps_between_clips,
      output_format,
   )
   return UCF101Dataset(ucf101)


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
   nonlin: Literal["relu", "gelu"]
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
   

class ActionRecognitionModel(nn.Module):

   def __init__(
      self,
      input_dims: tuple[int, int, int],
      num_classes: int,
      blocks: list[ModelBlock],
   ):
      super().__init__()

      conv_blks = []
      output_dims = input_dims
      for b in blocks:
         conv_blks.append(b.create_module())
         output_dims = b.compute_output_dim(output_dims)

      # FC
      fcs = [
         nn.Flatten(),
         nn.Linear(
            math.prod(output_dims) * blocks[-1].conv3d_params.out_channels,
            256
         ),
         nn.BatchNorm1d(num_features=256),
         nn.Linear(
            256,
            32,
         ),
         nn.BatchNorm1d(num_features=32),
         nn.Linear(
            32,
            num_classes
         ),
      ]
   
      self.arch = nn.ModuleList([
         *conv_blks,
         *fcs
      ])
      self._input_shape = (blocks[0].conv3d_params.in_channels, *input_dims)

   def print_sequential_arch(self, layer_num: int, layer: nn.Sequential, x: torch.Tensor):
      for i, submodule in enumerate(layer):
         in_shape = tuple(x.shape[1:])
         x = submodule(x)
         out_shape = tuple(x.shape[1:])
         print(f"├── {layer_num}.{i}: ({submodule.__class__.__name__}): {in_shape} -> {out_shape}")

   def print_arch(self):
      x = torch.zeros(2, *self._input_shape)  # use 2 b/c a batch of 1 breaks BatchNorm
      with torch.no_grad():
         for i, layer in enumerate(self.arch):
            in_shape = tuple(x.shape[1:])
            out = layer(x)
            out_shape = tuple(out.shape[1:])
            print(f"{i}: ({layer.__class__.__name__}): {in_shape} -> {out_shape}")
            
            # Print architecture within the Sequential module
            if isinstance(layer, nn.Sequential):
               self.print_sequential_arch(i, layer, x)

            x = out


   def forward(self, x: torch.Tensor):
      out = x
      for layer in self.arch:
         out = layer(out)
      return out
   

def train(
   dataset: IterableDataset,
   model: nn.Module,
   batch_size: int = 32,
   num_epochs: int = 10_000,
   lr=1e-3
):
   model.train()
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
   criterion = nn.CrossEntropyLoss()

   for epoch_num in range(num_epochs):
      training_step = 0
      running_loss = 0.0
      epoch_loss = 0.0
      
      for batch in dataloader:
         optimizer.zero_grad(set_to_none=True)

         # Forward pass
         videos, labels = batch[0], batch[1]
         out = model(videos)
         loss = criterion(out, labels)

         # Loss statistics
         running_loss += loss
         epoch_loss += loss

         # Backwards pass
         loss.backward()
         optimizer.step()

         # Average loss per epoch
         training_step += 1
         if training_step % 100 == 0:
            print(f"{training_step:5d} | avg loss (n=100): {(running_loss/100.0):.4f}")
            running_loss = 0.0
      
      print(f"epoch {epoch_num:.5d} | avg loss: {1.0 * epoch_loss/training_step}")


if __name__ == "__main__":
   model = ActionRecognitionModel(
      input_dims=(8, 240, 320),
      num_classes=10,
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
         )
      ]
   )

   # Model parameters
   print("================= MODEL ARCHITECTURE =================")
   model.print_arch()
   total_params = sum([p.numel() for p in model.parameters()])
   print(f"Total parameters: {total_params:,}")
   print("======================================================")

   ucf101 = load_ucf101(
      root=DATA / 'ucf101',
      annotation_path=DATA / 'annotations'
   )
   dataset = UCF101Dataset(ucf101)
   train(dataset, model)
