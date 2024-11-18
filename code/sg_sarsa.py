import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gymnasium import spaces
import plotly.graph_objects as go
import numpy as np

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

def q_hat(x, s, a, w):
  return x(s,a).dot(w)

def sg_greedy_action_selection(env, q_h, x, s, w, epsilon):
    rng = np.random.default_rng()
    if rng.random() > epsilon:
        return np.argmax([q_h(x, s, a, w) for a in range(env.action_space.n)])
    else:
        return env.action_space.sample()

def sg_sarsa(env, N: int, alpha: float, epsilon: float, x, w_shape):
    w = np.zeros(w_shape)
    q_0 = []

    for n in range(1, N):
        s = env.reset()
        done = False
        a = sg_greedy_action_selection(env, q_hat, x,  s, w, epsilon)
        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = sg_greedy_action_selection(env, q_hat, x, s_next, w, epsilon)
            if done:
              w = w + alpha * (r - q_hat(x, s, a, w))*x(s,a)
              break
            w = w + alpha * (r + 0.9 * q_hat(x, s_next, a_next, w) - q_hat(x, s, a, w))*x(s,a)
            s = s_next
            a = a_next

        q = np.max([q_hat(x, 0, a, w) for a in range(env.action_space.n)])
        q_0.append(q)

    return w, q_0

def eval(env, N, x, w):
    rewards = []
    for n in range(1, N):
        tot_reward = 0
        s = env.reset()
        done = False
        while not done:
            act = np.argmax([q_hat(x, s, a, w) for a in range(env.action_space.n)])
            s_next, r, done, _ = env.step(act)
            s = s_next
            tot_reward += r
        rewards.append(tot_reward)
    return rewards



def plot_conv(q_s0):
    plt.style.use('seaborn-v0_8')
    x = np.linspace(0, len(q_s0), len(q_s0))

    fig, ax1= plt.subplots()
    plt.subplots_adjust(hspace=0.5)

    ax1.plot(x, q_s0, linewidth=2.0, color="C1")
    ax1.set_title("Evolution of the value of the initial state")
    ax1.set_ylabel("Value of optimal action")
    ax1.set_xlabel("Episodes")

    plt.show()


def plot_v(w, x_f):

  # Create figure
  fig = go.Figure()

  x, y, text = [], [], []
  for i in range(4):
    for j in range(3):
      if j * 4 + i == 11:
        fig.add_shape(type="rect", x0=i, y0=j, x1=i + 1, y1=j + 1, line=dict(color="black"), fillcolor="green", layer="below")
      elif j * 4 + i == 7:
        fig.add_shape(type="rect", x0=i, y0=j, x1=i + 1, y1=j + 1, line=dict(color="black"), fillcolor="red", layer="below")
      elif j * 4 + i == 5:
        fig.add_shape(type="rect", x0=i, y0=j, x1=i + 1, y1=j + 1, line=dict(color="black"), fillcolor="gray", layer="below")
      else:
        fig.add_shape(type="rect", x0=i, y0=j, x1=i + 1, y1=j + 1, line=dict(color="black"), fillcolor="white", layer="below")
      x.append(i + 0.5)
      y.append(j + 0.5)
      text.append(int(np.max([q_hat(x_f, j * 4 + i, a, w)  for a in range(4)])))

      fig.add_trace(go.Scatter(
        x=x,
        y=y,
        text=text,
        mode="text",
        textfont=dict(
            color="black",
            size=18,
            family="Arial",
        ),
        visible=False
      )
  )


  fig.show()


if __name__ == '__main__':

    env = GridWorld()

    def f1(s,a):
        r = np.floor(s / 4)
        c = s % 4
        return np.array([1, r, c, a])

    def f2(s,a):
        r = np.floor(s / 4)
        c = s % 4
        x = np.array([1, r, c, 0, 0, 0, 0])
        x[3+a] = 1
        return x
    
    def f3(s,a):
      r = np.floor(s / 4)
      c = s % 4
      r_g = np.floor(11 / 4)
      c_g = 11 % 4
      d = np.abs(r - r_g) + np.abs(c - c_g)
      return np.array([1, r, c, a, d])
    
    def f4(s, a):
      r = np.floor(s / 4)  # Row of current state
      c = s % 4           # Column of current state
      r_g = np.floor(11 / 4)  # Row of the goal state
      c_g = 11 % 4           # Column of the goal state
      r_p = np.floor(7 / 4)  # Row of the penalty state
      c_p = 7 % 4           # Column of the penalty state
      
      d_goal = np.abs(r - r_g) + np.abs(c - c_g)  # Manhattan distance to the goal
      d_penalty = np.abs(r - r_p) + np.abs(c - c_p)  # Manhattan distance to the penalty
      
      # One-hot encoding for actions (0, 1, 2, 3 for up, down, left, right)
      action_encoding = np.zeros(4)
      action_encoding[a] = 1
      
      return np.concatenate([[1, r, c, d_goal, d_penalty], action_encoding])

    w, q_0 = sg_sarsa(env, 100000, 2e-13, 0.1, f4, 9)
    plot_conv(q_0)
    # plot_v(w, f3)
    # r = eval(env, 10000, f1, w)
    # plot_conv(r)

