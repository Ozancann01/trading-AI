import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class DQNAgent:
    def __init__(self, env, input_dim, action_size, device, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.action_size = action_size
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        output_dim = env.action_space.n

        self.q_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
  # Save the model to a file
    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    # Load the model from a file
    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.update_target_net()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
           return np.random.choice(self.action_size) 

        state = state.flatten()  # Add this line
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_tensor)
        return np.argmax(q_values.detach().cpu().numpy())


    def train(self, minibatch):
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array([s.flatten() for s in states])
        next_states = np.array([s.flatten() for s in next_states])

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.q_net(states_tensor).gather(1, actions_tensor)
        next_q_values = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
        target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)

        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay