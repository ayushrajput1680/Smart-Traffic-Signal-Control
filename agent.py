import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_sync_interval: int = 500,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.device = torch.device(device)
        self.step_counter = 0

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, epsilon: float = 0.05, valid_mask: np.ndarray = None) -> int:
        if random.random() < epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask == 1)[0]
                return int(np.random.choice(valid_actions))
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().squeeze(0)
        self.q_network.train()

        if valid_mask is not None:
            q_values[valid_mask == 0] = -1e9

        return int(np.argmax(q_values))

    def update(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())
