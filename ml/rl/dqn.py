"""
Deep-Q Learning
"""

import ale_py
from dataclasses import dataclass
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import math
import numpy as np
from pydantic import BaseModel, Field
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal



gym.register_envs(ale_py)


def print_metadata(env: gym.Env, prefix: str):
    log = [f"{prefix}"]
    log.append(f"n_actions: {env.action_space.n}")
    shape = env.observation_space.shape
    total_pixels = 1
    for d in shape:
        total_pixels *= d
    low, high = int(env.observation_space.low_repr), int(env.observation_space.high_repr)
    log.append(f"frame_dim: {shape}")
    log.append(f"total_pixels: {total_pixels}")
    log.append(f"pixel_vals: {low}-{high}")
    log.append(f"n_states {high-low+1}^{total_pixels}")
    print(" | ".join(log))


def make_env():
    # Make environment
    env = gym.make("ALE/SpaceInvaders-v5")
    print_metadata(env, prefix = "orig")

    # Apply following pre-processing steps to a frame:
    # 1. Reduce size to 84x84
    # 2. Convert pixels to grayscale
    # 3. Normalize pixels to be between 0 and 1
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(84,84))
    env = FrameStackObservation(env, stack_size=4)
    print_metadata(env, prefix = "proc")
    return env


@dataclass
class ReplayMemory:
    predicted_rewards: torch.Tensor = torch.Tensor([])
    next_states: torch.Tensor = torch.Tensor([])
    terminated: torch.Tensor = torch.Tensor([])

    @property
    def size(self):
        return self.next_states.shape[0]
    
    def add_step(self,
        predicted_reward: float,
        next_state: torch.Tensor,
        terminated: bool,
        truncated: bool
    ):
        # next_state is C x H x W
        if self.size == 0:
            self.next_states = torch.tensor(next_state).unsqueeze(0)
        else:
            self.next_states = torch.cat([
                self.next_states,
                torch.tensor(next_state).unsqueeze(0)
            ])  # B x C x H x W

        # Everything else is 1-D
        self.predicted_rewards = torch.cat([self.predicted_rewards, torch.tensor([predicted_reward])])
        self.terminated = torch.cat([self.terminated, torch.tensor([terminated or truncated])])

    def sample(self, n: int):
        idxs = torch.randint(0, self.size-1, (n,))
        return (
            self.predicted_rewards[idxs].unsqueeze(1),  # B x 1  
            self.next_states[idxs],  # B x C x H x W
            self.terminated[idxs].unsqueeze(1),  # B x 1  
        )


class TrainParams(BaseModel):
    replay_memory_capacity: int = Field(
        default=64,
        description="Number of episodes to store in replay memory."
    )
    batch_size: int = Field(
        default=32,
        description="Number of batches to sample from replay memory."
    )
    n_episodes: int = Field(
        default=1000,
        description="Number of episodes (full pass until termination) to use for training."
    )
    n_steps: int = Field(
        default=10000,
        description="Maximum number of steps in an episode."
    )
    n_steps_reset_action_model: int = Field(
        default=100,
        description="Update the action model to reflect the latest information from training."
    )
    min_eps: float = Field(
        default=0.05,
        description="Minimum probability for taking a random action."
    )
    max_eps: float = Field(
        default=1.0,
        description="Maximum probability for taking a random action."
    )
    gamma: float = Field(
        default=0.5,
        description="Discounting rate for future rewards."
    )
    lr: float = Field(
        default=0.7,
        description="Learning rate. Used for updating the Q-table."
    )

    def epsilon_decay(
        self,
        global_step: int,
        decay_type: Literal["linear", "cosine"]
    ):
        total_steps = self.n_episodes * self.n_steps

        if decay_type == "linear":
            slope = (self.max_eps - self.min_eps) / total_steps
            eps = self.max_eps - (slope * global_step)
            return max(self.min_eps, eps)
        else:
            # Adapted from PyTorch's CosineAnnealingLR
            # At step 1, assuming large number of total steps, this is the self.max_eps
            coeff = 0.5 * (1.0 + math.cos(math.pi * global_step / total_steps))
            return self.min_eps + coeff * (self.max_eps - self.min_eps)



class Conv2dParams(BaseModel):
    ch: int = Field(
        description="Number of channels."
    )
    out: int = Field(
        description="Number of convolutional filters with shape (ch * kernel * kernel)"
    )
    kernel: int = Field(
        description="Kernel size."
    )
    stride: int = Field(
        description="Stride of the convolution. Default is 1",
        default=1,
    )


class ModelParams(BaseModel):
    conv2d_params: list[Conv2dParams] = Field(
        default=[
            Conv2dParams(ch=32, out=64, kernel=8, stride=4),
            Conv2dParams(ch=64, out=64, kernel=4, stride=2),
            Conv2dParams(ch=64, out=64, kernel=4, stride=1),
        ]
    )
    mlp_params: int = 0

    def conv2d_output_channel_dims(self, frame_height: int, frame_width: int):
        output_dims = []
        for i, p in enumerate(self.conv2d_params):
            if i == 0:
                height, width = frame_height, frame_width
            else:
                height, width = output_dims[-1][0], output_dims[-1][1]

            if (height - p.kernel) % p.stride != 0:
                raise Exception(f"ERROR | frame height: {height} | kernel: {p.kernel} | stride: {p.stride}")
            if (width - p.kernel) % p.stride != 0:
                raise Exception(f"ERROR | frame width: {width} | kernel: {p.kernel} | stride: {p.stride}")
            
            dimh = (height - p.kernel) / p.stride + 1
            dimw = (width - p.kernel) / p.stride + 1
            output_dims.append((p.out, dimh, dimw))
        
        return output_dims


class InputParams(BaseModel):
    channels: int = Field(
        default=4,
        description="Number of frames to stack for capturing temporal effects."
    )
    height: int = Field(
        default=84,
        description="Height of frame (in pixels)."
    )
    width: int = Field(
        default=84,
        description="Width of frame (in pixels)"
    )


class QFunctionParams(BaseModel):
    train: TrainParams = Field(
        default_factory=TrainParams
    )
    input: InputParams = Field(
        default_factory=InputParams
    )
    n_actions: int = Field(
        default=4,
        description="Number of actions in our environment."
    )


class QFunction(nn.Module):

    def __init__(self, params: QFunctionParams):
        super().__init__()
        self.params = params
        self.conv = [
            # width = 84 --> (84 - 8)/4 + 1 = 20
            # height = 84 --> (84 - 8)/4 + 1 = 20
            nn.Conv2d(in_channels=params.input.channels, out_channels=32, kernel_size=8, stride=4),  # B x 32 x 20 x 20
            nn.ReLU(),
            # width = 20 --> (20 - 4)/2 + 1 = 9
            # height = 20 --> (20 - 4)/2 + 1= 9
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # B x 64 x 9 x 9
            nn.ReLU(),
            # width = 9 --> (9 - 4)/1 + 1 = 6
            # height = 9 --> (9 - 4)/1 + 1 = 6
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1),  # B x 64 x 6 x 6
            nn.ReLU(),
        ]
        
        # Number of input channels for first linear layer of MLP.
        # Convolutional layers outputs tensor of shape (C, H, W)
        # C = number of channels
        # H = height of output map
        # W = width of output map
        # We flatten this to a 1-D tensor
        linear_in = 64 * 6 * 6
        self.mlp = [
            nn.Flatten(),
            nn.Linear(int(linear_in), 512),
            nn.ReLU(),
            nn.Linear(512, params.n_actions),
        ]
        self.layers = nn.ModuleList((self.conv + self.mlp))
    
    def forward(self, state: torch.Tensor):
        # State has dim (B, C, H, W)
        # B = batch size
        # C = channels (number of stacked frames)
        # H = height of frame
        # W = width of frame

        # Layers outputs the rewards from each action
        out = state.to(torch.float32)
        for layer in self.layers:
            out = layer(out)
        return out


def choose_action(env: gym.Env, model: QFunction, state: np.ndarray, eps: float):
    if random.random() < eps:
        return env.action_space.sample()
    else:
        rewards = model(state).squeeze()
        return np.argmax(np.array(rewards))


def train(
    env: gym.Env,
    model: QFunction,
    action_model: QFunction,
    optim: torch.optim.AdamW
):
    params = model.params
    replay_memory = ReplayMemory()
    global_step = 0
    for n in range(params.train.n_episodes):
        training_step = 0
        state, _ = env.reset()
        optim.zero_grad(set_to_none=True)

        for step in range(params.train.n_steps):
            eps = params.train.epsilon_decay(
                global_step,
                "linear"
            )
            action = choose_action(env, model, state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_memory.add_step(
                reward,
                next_state,
                terminated,
                truncated
            )

            # Sample from replay memory for training
            if replay_memory.size > params.train.batch_size:
                (
                    rewards,
                    next_states,
                    terminateds,
                ) = replay_memory.sample(params.train.batch_size)
                
                # Discounted future rewards
                expected_future_rewards = action_model(next_states).max(dim=1, keepdim=True).values  # B x 1
                expected_future_rewards.masked_fill_(terminateds == True, 0.0)  # replace with 0 if terminated
                total_rewards = rewards + params.train.gamma * expected_future_rewards  # discount

                # Backward
                loss = F.mse_loss(rewards, total_rewards)

                if (global_step+1) % 100 == 0:
                    print(f"step {global_step} | loss {loss:.4f}")

                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

                # Update action model
                if (training_step+1) % params.train.n_steps_reset_action_model == 0:
                    for k, v in model.state_dict().items():
                        action_model.state_dict()[k] = v
                    training_step = 0
                
                training_step += 1

            global_step += 1


if __name__ == "__main__":
    env = make_env()
    p = QFunctionParams()
    train_model = QFunction(p)

    # Action model should be equivalent to train model.
    action_model = QFunction(p)
    for k, s in train_model.state_dict().items():
        action_model.state_dict()[k] = s
    
    # Optimizer
    optim = torch.optim.AdamW(params=train_model.parameters(), lr=p.train.lr, betas=(0.9, 0.95), eps=1e-8)
    state, _ = env.reset()
    train(env, train_model, action_model, optim)
