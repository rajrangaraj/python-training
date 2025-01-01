"""
Demonstration of reinforcement learning using OpenAI Gym and Q-learning.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    env_name: str = "CartPole-v1"
    episodes: int = 1000
    max_steps: int = 500
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    """Experience replay memory for DQN."""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Save a transition."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """DQN Agent implementation."""
    
    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = config.epsilon_start
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(config.device)
        self.target_net = DQN(state_size, action_size).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        
        # Memory
        self.memory = ReplayMemory(config.memory_size)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                state_tensor = state_tensor.to(self.config.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).to(self.config.device)
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def train_agent(config: RLConfig) -> Tuple[List[float], List[float]]:
    """Train DQN agent."""
    
    # Create environment
    env = gym.make(config.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(config, state_size, action_size)
    
    # Training metrics
    episode_rewards = []
    losses = []
    
    # Training loop
    for episode in range(config.episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        
        for step in range(config.max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train model
            loss = agent.train_step()
            episode_loss += loss
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % config.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        # Decay epsilon
        agent.epsilon = max(
            config.epsilon_end,
            agent.epsilon * config.epsilon_decay
        )
        
        # Record metrics
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / (step + 1))
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{config.episodes}")
            print(f"Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
            print(f"Epsilon: {agent.epsilon:.2f}")
    
    env.close()
    return episode_rewards, losses

def visualize_training(rewards: List[float], losses: List[float]):
    """Visualize training progress."""
    
    plt.figure(figsize=(12, 4))
    
    # Rewards plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    return plt

def demonstrate_trained_agent(agent: DQNAgent, config: RLConfig):
    """Demonstrate trained agent's performance."""
    
    env = gym.make(config.env_name, render_mode='human')
    state = env.reset()
    total_reward = 0
    
    for _ in range(config.max_steps):
        env.render()
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    env.close()
    return total_reward

if __name__ == '__main__':
    # Configuration
    config = RLConfig()
    
    # Train agent
    rewards, losses = train_agent(config)
    
    # Visualize results
    training_plot = visualize_training(rewards, losses)
    training_plot.show() 