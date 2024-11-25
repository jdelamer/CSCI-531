import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class GridWorld(gymnasium.Env):
  def __init__(self):
    # Define the action and observation spaces
    self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
    self.observation_space = spaces.Discrete(12) # 12 cells
    # Initialize the state
    self.state = 0
    self.terminals = [11, 7]

  def step(self, action: int):

    self._transition(action)
    done = False
    reward = 0
    if self.state == 11:
      reward = 10
      done = True
    elif self.state == 7:
      reward = -10
      done = True
    # Return the observation, reward, done flag, and info
    return self.state, reward, done, {}

  def _transition(self, action: int):
    """
    Transition function.
    :param action: Action to take
    """
    r = np.floor(self.state / 4)
    c = self.state % 4

    prob = np.random.random()
    if prob < 0.80:
      actual_action = action
    elif prob < 0.90:
      # Adjacent cell "clockwise"
      actual_action = (action + 1) % 4
    else:
      # Adjacent cell "counter clockwise"
      actual_action = (action - 1) % 4

    if actual_action == 0:
      r = max(0, r - 1)
    elif actual_action == 2:
      r = min(2, r + 1)
    elif actual_action == 1:
      c = max(0, c - 1)
    elif actual_action == 3:
      c = min(3, c + 1)
    self.state = int(r * 4 + c)

  def reset(self):
    """
    Reset the environment.
    """
    self.state = 0
    return self.state

  def render(self, render="human"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')


    for i in range(4):
      for j in range(3):
        if j * 4 + i == 11:
          rect = Rectangle((i, j), 1, 1, edgecolor='black', facecolor='green')
          ax.add_patch(rect)
        elif j * 4 + i == 7:
          rect = Rectangle((i, j), 1, 1, edgecolor='black', facecolor='red')
          ax.add_patch(rect)
        elif j * 4 + i == 5:
          rect = Rectangle((i, j), 1, 1, edgecolor='black', facecolor='grey')
          ax.add_patch(rect)
        else:
          rect = Rectangle((i, j), 1, 1, edgecolor='black', facecolor='white')
          ax.add_patch(rect)

    ax.tick_params(axis='both',       # changes apply to both axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False) # labels along the bottom edge are off

    plt.show()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)
    
def compute_loss(batch, policy_net, target_net, gamma):
    states, actions, rewards, next_states = batch

    # Compute current Q values
    current_q_values = policy_net(states).gather(1, actions)

    # Compute target Q values
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + gamma * max_next_q_values

    # Compute MSE loss
    loss = nn.MSELoss()(current_q_values, target_q_values)
    return loss

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*samples)
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).unsqueeze(1)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)
        
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 buffer_size=10000, batch_size=64, target_update=100):

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        self.steps = 0

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(np.array([state]))
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)

        # Compute loss
        loss = compute_loss(batch, self.policy_net, self.target_net, self.gamma)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_epsilon()

def dqn(env, episodes=1000):
    state_dim = 1
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state)
            
            # Train the network
            loss = agent.train()
            
            total_reward += reward
            state = next_state
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")


env = GridWorld()
dqn(env, 100000)