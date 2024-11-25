---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Deep Q-Learning

Deep Q-Learning (DQN) combines Q-learning with deep neural networks to handle high-dimensional state spaces. This approach was popularized by DeepMind's breakthrough paper where they achieved human-level performance on Atari games using only raw pixel inputs.

## From Q-Learning to Deep Q-Learning

In traditional Q-learning, we maintain a Q-table that stores Q-values for each state-action pair. However, this becomes impractical for large state spaces. Deep Q-Learning addresses this by approximating the Q-function using a neural network:

```{math}
Q(s,a;\theta) \approx Q^*(s,a)
```

where $\theta$ represents the neural network parameters.

### Network Architecture

The DQN takes a state as input and outputs Q-values for all possible actions:

```{image} ./img/dqn_architecture.png
:align: center
:width: 80%
```

````{prf:example} Environment with 4 actions

For example, in an environment with 4 possible actions, the network would output:

```{math}
\begin{bmatrix} Q(s,a_1;\theta) \\ Q(s,a_2;\theta) \\ Q(s,a_3;\theta) \\ Q(s,a_4;\theta) \end{bmatrix}
```

````

### Algorithm

In Q-learning, the agent uses its experiences to update its action-value function as follows:

```{math}
Q(s_t,a_t) = Q(s_t,a_t) + \alpha\left( r(s,a) + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
```

However, in DRL we update the parameters $\theta$ instead. The loss function will be defined again as the MSE, with the error being the difference between the target value $y_t$ and the current estimation such as:

```{math}
\mathcal{L}(\theta) = \left(y_t - Q(s_t, a_t; \theta)\right)^2
```

with the target $y_t$ being:

```{math}
y_t =
\begin{cases}
r_t & \text{if } s_{t+1} \text{ is terminal} \\
r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta) & \text{otherwise}
\end{cases}
```

````{prf:algorithm} DQN
:label: alg:dqn

$
\begin{array}{l}
\text{Initialize value network}\ Q\ \text{with random parameters}\ \theta\\
\textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
\quad\quad \text{Observe}\ s\\
\quad\quad a \leftarrow \epsilon\text{-greedy}\\
\quad\quad s',r \leftarrow \text{Execute } a\\
\quad\quad \textbf{If}\ s'\ \text{terminal }\textbf{then}\\
\quad\quad\quad\quad y \leftarrow r\\
\quad\quad \textbf{Else}\\
\quad\quad\quad\quad y \leftarrow r + \gamma\max_{a'} Q(s',a';\theta)\\
\quad\quad \mathcal{L}(\theta) \leftarrow \left(y - Q(s, a; \theta)\right)^2\\
\quad\quad \text{Update parameters}\ \theta\ \text{by minimizing}\ \mathcal{L}(\theta)
\end{array}
$

````

This algorithm was innovating, but had two main issues, the moving target problem and the problem of correlations.

## Key Innovations

### Target Network

Another innovation is the use of a separate target network for computing target Q-values, which helps solve the "moving target" problem. When we update our Q-network, we're chasing a target that is itself moving since it depends on the same network we're updating. This creates instability in training.

To address this, we use a separate target network:

```{math}
y_i = r_i + \gamma \max_{a'} Q(s_{i+1}, a';\theta^-)
```

where $\theta^-$ represents the parameters of the target network that are periodically copied from the main network.

We can now rewrite the loss function to consider the target network:

```{math}
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\left[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\right]
```

### Experience Replay

DQN introduced experience replay, which addresses the problem of correlated samples in sequential data. When learning from consecutive samples of experience, the strong correlations between subsequent states can create inefficient learning and cause the network to get stuck in local optima. To break these correlations, DQN stores transitions in a replay buffer:

```{prf:definition} Replay Buffer
A replay buffer $\mathcal{D}$ stores transitions $(s_t, a_t, r_t, s_{t+1})$ from which we can sample randomly for training.
```

This random sampling breaks the temporal correlations and provides more independent and identically distributed (i.i.d.) training data, which leads to more stable learning.

Replay buffers are very simple to implement:

```{code-cell} ipython3
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

## Training Algorithm

Including these two improvments we can rewrite the algorithm.

````{prf:algorithm} DQN with improvments
:label: alg:dqn2

$
\begin{array}{l}
\text{Initialize value network}\ Q\ \text{with random parameters}\ \theta\\
\text{Initialize target network with parameters}\ \theta^- = \theta\\
\text{Initialize an empty replay buffer}\ \mathcal{D}=\{\}\\
\textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
\quad\quad \text{Observe}\ s\\
\quad\quad a \leftarrow \epsilon\text{-greedy using } Q(s,a;\theta)\\
\quad\quad s',r \leftarrow \text{Execute } a\\
\quad\quad \text{Store transition}\ (s,a,r,s')\ \text{in replay buffer} \mathcal{D}\\
\quad\quad \text{Sample random mini-batch of}\ B\ \text{transitions}\ (s_k,a_k,r_k,s_{k+1})\in \mathcal{D}\\
\quad\quad \textbf{If}\ s_{k+1}\ \text{terminal }\textbf{then}\\
\quad\quad\quad\quad y_k \leftarrow r_k\\
\quad\quad \textbf{Else}\\
\quad\quad\quad\quad y_k \leftarrow r_k + \gamma\max_{a'} Q(s_{k+1},a';\theta^-)\\
\quad\quad \mathcal{L}(\theta) \leftarrow \frac{1}{B}\sum_{k=1}^B\left(y_k - Q(s_k, a_k; \theta)\right)^2\\
\quad\quad \text{Update parameters}\ \theta\ \text{by minimizing}\ \mathcal{L}(\theta)\\
\quad\quad \text{Every set interval, update target network parameters}\ \theta^-
\end{array}
$

````

## Implementation

First we can implement our neural network using the library PyTorch.

```{code-cell} ipython3
import torch
import torch.nn as nn

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
```

The loss function is simple, but takes our two networks, the target and the policy.

```{code-cell} ipython3
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
```

The replay buffer as seen previously, but using PyTorch instead.

```{code-cell} ipython3
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
```

Usually, we create an agent that will contain the NN and all the functions for the learning itself. We separate it from the global loop of the algorithm to differentiate the learning part and the sample collection.

```{code-cell} ipython3
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

```

Finally, we have the algorithm itself.

```{code-cell} ipython3
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
```

## Enhancements

Several enhancements to the basic DQN algorithm have been proposed:

1. **Double DQN**: Reduces overestimation of Q-values
2. **Dueling DQN**: Separates value and advantage estimation
3. **Prioritized Experience Replay**: Samples important transitions more frequently

````{prf:example} Double DQN
Instead of computing target values as:
```{math}
y_i = r_i + \gamma \max_{a'} Q(s_{i+1}, a';\theta^-)
```
Double DQN uses:
```{math}
y_i = r_i + \gamma Q(s_{i+1}, \arg\max_{a'} Q(s_{i+1}, a';\theta);\theta^-)
```
````

## Performance Considerations

When implementing DQN, several factors affect performance:

1. **Network Architecture**: Deeper networks aren't always better
2. **Hyperparameters**: Buffer size, batch size, update frequency
3. **Pre-processing**: State normalization, reward scaling
4. **Hardware**: GPU acceleration for larger networks

```{note}
DQN typically requires significant computational resources and careful tuning to achieve good performance.
```
