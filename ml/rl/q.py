"""
Q-learning
"""

import gymnasium as gym
import numpy as np
import random
from typing import Literal
from pydantic import BaseModel, Field
import math


# Make FrozenLake environment
def make_env():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    state_space = env.observation_space
    action_space = env.action_space
    print(f"There are {state_space.n} possible states.")
    print(f"There are {action_space.n} possible actions.")
    return env


def initialize_q_table(n_states: int, n_actions: int) -> np.ndarray:
    return np.zeros((n_states, n_actions))


def greedy_policy(state: int, q_table: np.ndarray) -> int:
    """
    Given a state, take the action that has the highest reward.
    """
    return np.argmax(q_table[state])


def epsilon_greedy_policy(state: int, action_space: gym.Space, q_table: np.ndarray, epsilon: float):
    """
    Given a state, either:
    - Take a random action with probability epsilon, or
    - Take the action that has the highest reward
    """
    p_random = random.random()
    if p_random < epsilon:
        return action_space.sample()
    return greedy_policy(state, q_table)


def epsilon_decay(
    step: int,
    total_steps: int,
    decay_type: Literal["linear", "cosine"],
    max_eps: float = 1.0,
    min_eps: float = 0.05
):
    """
    Epsilon is our "exploration" probability. When we first start, we want this to be close to 1,
    because our Q table doesn't have much information. However, as we learn, we want this to decrease
    steadily.
    """
    if decay_type == "linear":
        # At step 1, we want our epsilon to be the max_eps
        eps = max_eps - (step-1) * (max_eps - min_eps)/total_steps
        return max(min_eps, eps)
    elif decay_type == "cosine":
        # Adapted from PyTorch's CosineAnnealingLR
        # At step 1, assuming large number of total steps, this is the max_eps
        coeff = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
        return min_eps + coeff * (max_eps - min_eps)
    else:
        raise Exception(f"Unknown decay_type={decay_type}. Expected 'linear' or 'cosine'.")


class TrainingHyperparams(BaseModel):
    n_episodes: int = Field(
        default=10000,
        description="Number of episodes to use for training"
    )
    lr: float = Field(
        default=0.7,
        description="Learning rate. Used for updating the Q-table."
    )
    n_steps: int = Field(
        default=99,
        description="Maximum number of steps in an episode."
    )
    gamma: float = Field(
        default=0.5,
        description="Discounting rate for future rewards."
    )
    max_eps: float = Field(
        default=1.0,
        description="Exploration probability at the start of learning."
    )
    min_eps: float = Field(
        default=0.05,
        description="Minimum exploration probability."
    )
    n_eval_episodes: int = Field(
        default=100,
        descpription="Number of episodes to use to evaluate the agent during training."
    )


def evaluate_agent(params: TrainingHyperparams, q_table: np.ndarray, env: gym.Env):
    episode_rewards = []
    for ep in range(params.n_eval_episodes):
        state, _ = env.reset()

        truncated, terminated = False, False
        total_rewards_ep = 0

        for _ in range(params.n_steps):
            action = greedy_policy(state, q_table)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            
            state = new_state

        episode_rewards.append(total_rewards_ep)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def train(params: TrainingHyperparams, q_table: np.ndarray, env: gym.Env):
    
    global_step = 0
    for ep in range(params.n_episodes):
        state, _ = env.reset()

        for _ in range(params.n_steps):
            global_step += 1

            # Epsilon for controlling randomness of actions.
            eps = epsilon_decay(
                global_step,
                params.n_episodes * params.n_steps,
                "linear",
                params.max_eps,
                params.min_eps
            )
            action = epsilon_greedy_policy(state, env.action_space, q_table, eps)
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-Table
            predicted_reward = q_table[state][action]

            # Expected reward = current reward + discounted future reward
            # Future reward is calculated with best next-state action value, i.e., greedy policy when in next_state
            # This is called off-policy: the acting policy is greedy-epsilon but the updating policy is greedy
            expected_reward = reward + params.gamma * q_table[new_state][greedy_policy(new_state, q_table)]
            q_table[state][action] = predicted_reward + params.lr * (expected_reward - predicted_reward)

            if truncated or terminated:
                break

            state = new_state

        # Print some evaluation stats
        if ep > 0 and ep % 100 == 0:
            mean_reward, std_reward = evaluate_agent(params, q_table, env)
            print(f"episode {ep} | mean reward: {mean_reward:.4f} | stddev reward: {std_reward:.4f}") 


if __name__ == "__main__":
    env = make_env()
    q_table = initialize_q_table(env.observation_space.n, env.action_space.n) 
    params = TrainingHyperparams()
    train(params, q_table, env)
