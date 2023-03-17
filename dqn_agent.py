import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        self.q_net = QNetwork(input_dim, output_dim)
        self.target_net = QNetwork(input_dim, output_dim)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # Other methods for training, selecting actions, and managing the replay buffer will be added here
def select_action(self, state, eval_mode=False):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = self.q_net(state_tensor)
    action = torch.argmax(q_values).item()

    if not eval_mode and np.random.rand() < self.epsilon:
        action = self.env.action_space.sample()

    return action


def update_network(self, experiences):
    states, actions, rewards, next_states, dones = zip(*experiences)

    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions).unsqueeze(1)
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.BoolTensor(dones).unsqueeze(1)

    current_q_values = self.q_net(states_tensor).gather(1, actions_tensor)
    next_q_values = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
    target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)

    loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
