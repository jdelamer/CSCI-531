import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample


class CliffWalkingEnv(Env):
    """
    Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.

    ## Description
    The game starts with the player at location [3, 0] of the 4x12 grid world with the
    goal located at [3, 11]. If the player reaches the goal the episode ends.

    A cliff runs along [3, 1..10]. If the player moves to a cliff location it
    returns to the start location.

    The player makes moves until they reach the goal.

    Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].

    The cliff can be chosen to be slippery (disabled by default) so the player may move perpendicular
    to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).

    With inspiration from:
    [https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left

    ## Observation Space
    There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at
    the goal as the latter results in the end of the episode. What remains are all
    the positions of the first 3 rows plus the bottom-left cell.

    The observation is a value representing the player's current position as
    current_row * ncols + current_col (where both the row and col start at 0).

    For example, the starting position can be calculated as follows: 3 * 12 + 0 = 36.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[36]` (location [3, 0]).

    ## Reward
    Each time step incurs -1 reward, unless the player stepped into the cliff,
    which incurs -100 reward.

    ## Episode End
    The episode terminates when the player enters state `[47]` (location [3, 11]).

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - "p" - transition proability for the state.

    As cliff walking is not stochastic, the transition probability returned always 1.0.

    ## References
    <a id="cliffwalk_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    """

    

    def __init__(self, is_slippery: bool = False):
        self.shape = (4, 12)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.is_slippery = is_slippery

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        self.POSITION_MAPPING = {self.UP: [-1, 0], self.RIGHT: [0, 1], self.DOWN: [1, 0], self.LEFT: [0, -1]}

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][self.UP] = self._calculate_transition_prob(position, self.UP)
            self.P[s][self.RIGHT] = self._calculate_transition_prob(position, self.RIGHT)
            self.P[s][self.DOWN] = self._calculate_transition_prob(position, self.DOWN)
            self.P[s][self.LEFT] = self._calculate_transition_prob(position, self.LEFT)

        # Calculate initial state distribution
        # We always start in state (3, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)


    def _limit_coordinates(self, coord):
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, move: int):
        """Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition

        Returns:
            Tuple of ``(transition_probability, new_state, reward, terminated)``
            where `transition_probability` is 1 if the environment is not slippery, otherwise 1/3 for `move`
            and the perpendicular moves.
        """
        if not self.is_slippery:
            deltas = [self.POSITION_MAPPING[move]]
        else:
            deltas = [
                self.POSITION_MAPPING[act] for act in [(move - 1) % 4, move, (move + 1) % 4]
            ]
        outcomes = []
        for delta in deltas:
            new_position = np.array(current) + np.array(delta)
            new_position = self._limit_coordinates(new_position).astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            if self._cliff[tuple(new_position)]:
                outcomes.append((1 / len(deltas), self.start_state_index, -100, False))
            else:
                terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
                is_terminated = tuple(new_position) == terminal_state
                if is_terminated:
                   outcomes.append((1 / len(deltas), new_state, 0, is_terminated))
                else:
                    outcomes.append((1 / len(deltas), new_state, -1, is_terminated))
        return outcomes

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        _, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        return int(s), r, t

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        return int(self.s)



def value_iteration(env, gamma=0.9, theta=0.0001):
  V = np.zeros(env.observation_space.n)
  while True:
    delta = 0
    for s in range(env.observation_space.n):
      v = V[s]
      V[s] = max([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in env.P[s]])
      delta = max(delta, np.abs(v - V[s]))
    if delta < theta:
      break
  pi = np.zeros(env.observation_space.n)
  for s in range(env.observation_space.n):
    pi[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in env.P[s]])
  return V, pi

def run(env, pi):
    """
    Run the policy on the environment and returns the cumulative reward.
    :param: env: The environment
    :param: pi: The policy calculated by value iteration
    :return: Cumulative reward
    """
    s = env.reset()
    done = False
    sum_r = 0
    while not done:
        a = pi[s]
        s, r, done = env.step(a)
        sum_r += r
    return sum_r

def evaluate(env, pi, N):
    """
    Evaluate the policy N times on the environment.
    """
    cumulated_rewards = 0
    for _ in range(N):
        cumulated_rewards += run(env, pi)
    return cumulated_rewards / N


env_false = CliffWalkingEnv(False)
v, pi = value_iteration(env_false, theta=0.001)


env_true = CliffWalkingEnv(True)
v, pi = value_iteration(env_false, theta=0.001)