#!/usr/bin/env python3
"""
Monte Carlo Gridworld Visual Demonstration

This script implements a simple episodic Gridworld with:
- a Start cell
- a Goal cell (reward +1, terminal)
- a Hole cell (reward -1, terminal)

It runs First-Visit On-Policy Monte Carlo Control (epsilon-greedy)
and updates visualizations in real time:
- Left: heatmap of state values V(s) = max_a Q(s,a) and greedy policy arrows
- Right: plot of V(start) across episodes (moving average)

Usage:
    python slides/examples/mc_grid_visual.py

Requires:
    numpy, matplotlib

Notes for students:
- The on-policy algorithm learns action values Q(s,a) from episodic returns.
- We use epsilon-greedy action selection and random tie-breaking to ensure exploration.
- The visualization updates every `update_interval` episodes to keep UI responsive.
- Convergence to 0.5 is expected due to: optimistic initialization, discount factor,
  exploration noise, and distance from terminals affecting discounted rewards.
"""

import random
import time
from collections import defaultdict, deque
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Gridworld environment
# -------------------------


class GridWorld:
    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None,
        hole: Tuple[int, int] = None,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # default positions
        self.start = start if start is not None else (n_rows - 1, 0)
        self.goal = goal if goal is not None else (0, n_cols - 1)
        self.hole = hole if hole is not None else (1, 1)
        self.state = self.start

        # Actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
        self.actions = [0, 1, 2, 3]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action: int):
        r, c = self.state
        if action == 0:
            r2, c2 = max(0, r - 1), c
        elif action == 1:
            r2, c2 = r, min(self.n_cols - 1, c + 1)
        elif action == 2:
            r2, c2 = min(self.n_rows - 1, r + 1), c
        elif action == 3:
            r2, c2 = r, max(0, c - 1)
        else:
            raise ValueError("Invalid action")

        # Move
        self.state = (r2, c2)

        # Terminal checks
        if self.state == self.goal:
            return self.state, 1.0, True
        if self.state == self.hole:
            return self.state, -1.0, True
        return self.state, 0.0, False

    def state_space(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        return s == self.goal or s == self.hole


# -------------------------
# Monte Carlo Control
# -------------------------


def epsilon_greedy_action(
    Q: Dict[Tuple[Tuple[int, int], int], float],
    state: Tuple[int, int],
    actions: List[int],
    epsilon: float,
) -> int:
    """Epsilon-greedy with random tie-breaking among maxima."""
    if random.random() < epsilon:
        return random.choice(actions)
    q_vals = [Q[(state, a)] for a in actions]
    max_val = max(q_vals)
    max_actions = [a for a, q in zip(actions, q_vals) if q == max_val]
    return random.choice(max_actions)


def generate_episode(
    env: GridWorld,
    policy_fn,
    start_state: Tuple[int, int] = None,
    start_action: int = None,
) -> List[Tuple[Tuple[int, int], int, float]]:
    """
    Generates an episode following policy_fn. If start_state/start_action are provided,
    the episode begins from that state and first action; otherwise resets to env.start.
    Returns list of (s,a,r).
    """
    episode = []
    if start_state is not None:
        # start from the provided start_state (exploring starts)
        env.state = start_state
        a = start_action if start_action is not None else policy_fn(start_state)
        s2, r, done = env.step(a)
        episode.append((start_state, a, r))
        if done:
            return episode
    else:
        # normal start
        env.reset()
    while True:
        s = env.state
        a = policy_fn(s)
        s2, r, done = env.step(a)
        episode.append((s, a, r))
        if done:
            break
    return episode


def first_visit_mc_control(
    env: GridWorld,
    num_episodes: int = 2000,
    gamma: float = 1.0,
    epsilon: float = 0.1,
):
    """
    First-Visit Monte Carlo Control (on-policy, epsilon-greedy).
    Returns Q, policy, and a list of V(start) estimates per episode.
    """
    actions = env.actions
    Q = defaultdict(float)  # (s,a) -> value
    Returns = defaultdict(list)  # (s,a) -> list of returns
    V_start_history = []

    def policy_fn(s):
        return epsilon_greedy_action(Q, s, actions, epsilon)

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(env, policy_fn)
        G = 0.0
        visited_sa = set()
        # traverse backwards to compute returns
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                Returns[(s, a)].append(G)
                Q[(s, a)] = float(np.mean(Returns[(s, a)]))
        # record V(start) = max_a Q(start,a)
        q_vals = [Q[((env.start), a)] for a in actions]
        V_start = float(max(q_vals))
        V_start_history.append(V_start)
    # derive greedy policy
    policy = {}
    for s in env.state_space():
        if env.is_terminal(s):
            continue
        q_vals = [Q[(s, a)] for a in actions]
        max_val = max(q_vals)
        max_actions = [a for a, q in zip(actions, q_vals) if q == max_val]
        policy[s] = random.choice(max_actions)
    return Q, policy, V_start_history


# -------------------------
# Visualization utilities
# -------------------------


def compute_state_values_from_Q(
    env: GridWorld, Q: Dict[Tuple[Tuple[int, int], int], float]
) -> np.ndarray:
    """Return a 2D array of V(s) = max_a Q(s,a) for non-terminals; terminals get their rewards."""
    V = np.zeros((env.n_rows, env.n_cols))
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)
            if s == env.goal:
                V[r, c] = 1.0
            elif s == env.hole:
                V[r, c] = -1.0
            else:
                q_vals = [Q[(s, a)] for a in env.actions]
                V[r, c] = float(max(q_vals))
    return V


def plot_grid_and_policy(ax, env: GridWorld, Q: Dict, policy: Dict):
    """Plot heatmap of V(s) and greedy policy arrows."""
    V = compute_state_values_from_Q(env, Q)
    im = ax.imshow(V, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("State values V(s) and greedy policy")
    ax.set_xticks(np.arange(env.n_cols))
    ax.set_yticks(np.arange(env.n_rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # annotate start/goal/hole
    sr, sc = env.start
    gr, gc = env.goal
    hr, hc = env.hole
    # Draw markers
    ax.text(
        sc,
        sr,
        "S",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        gc,
        gr,
        "G",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        hc,
        hr,
        "H",
        ha="center",
        va="center",
        color="black",
        fontsize=12,
        fontweight="bold",
    )

    # Draw policy arrows
    X, Y, U, Vv = [], [], [], []
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)
            if env.is_terminal(s):
                # no arrow
                continue
            a = policy.get(s, None)
            if a is None:
                continue
            # arrow direction: up(0), right(1), down(2), left(3)
            dx, dy = 0.0, 0.0
            if a == 0:
                dy = -0.4
            elif a == 1:
                dx = 0.4
            elif a == 2:
                dy = 0.4
            elif a == 3:
                dx = -0.4
            # matplotlib's imshow: x is columns, y is rows
            X.append(c)
            Y.append(r)
            U.append(dx)
            Vv.append(dy)
    if X:
        ax.quiver(X, Y, U, Vv, angles="xy", scale_units="xy", scale=1.0, color="k")


def plot_value_line(ax, V_history: List[float], window: int = 50):
    """Plot V(start) over episodes with moving average and convergence info."""
    ax.clear()
    ax.set_title("V(start) over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("V(start)")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    episodes = np.arange(1, len(V_history) + 1)
    ax.plot(episodes, V_history, color="gray", alpha=0.3, label="V(start)")
    if len(V_history) >= window:
        ma = np.convolve(V_history, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, len(V_history) + 1),
            ma,
            color="blue",
            label=f"{window}-ep MA",
        )
        # Show convergence value
        current_value = ma[-1] if len(ma) > 0 else V_history[-1]
        ax.axhline(
            y=current_value,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Current: {current_value:.3f}",
        )
    ax.legend()


# -------------------------
# Main interactive demo
# -------------------------


def run_visual_demo(
    n_rows: int = 5,
    n_cols: int = 5,
    start: Tuple[int, int] = None,
    goal: Tuple[int, int] = None,
    hole: Tuple[int, int] = None,
    num_episodes: int = 2000,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    update_interval: int = 25,
    alpha: float = 0.1,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    optimistic_init: float = 0.0,
    exploring_starts_prob: float = 0.1,
):
    env = GridWorld(
        n_rows=n_rows,
        n_cols=n_cols,
        start=start,
        goal=goal,
        hole=hole,
    )

    # initialize Q and policy (optimistic initialization helps exploration)
    optimistic = optimistic_init
    Q = defaultdict(lambda: optimistic)

    print(f"Demo Parameters:")
    print(
        f"  Grid: {n_rows}x{n_cols}, Start: {env.start}, Goal: {env.goal}, Hole: {env.hole}"
    )
    print(f"  Gamma: {gamma}, Epsilon: {epsilon}, Episodes: {num_episodes}")
    print(f"  Optimistic Init: {optimistic_init}")
    print(
        f"  Expected V(start) depends on shortest path, discount factor, and exploration"
    )
    Returns = defaultdict(list)
    Counts = defaultdict(int)
    policy = {}  # greedy policy for plotting
    V_start_history = []

    # prepare plots
    plt.ion()
    fig, (ax_grid, ax_line) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Monte Carlo Control: Gridworld Visual Demo", fontsize=14)

    # prefill with zeros so the color scale is meaningful
    V = compute_state_values_from_Q(env, Q)
    im = ax_grid.imshow(V, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plot_value_line(ax_line, V_start_history)

    # show once
    plt.draw()
    plt.pause(0.1)

    actions = env.actions

    def policy_fn(s):
        return epsilon_greedy_action(Q, s, actions, epsilon)

    try:
        # initialize decaying epsilon for exploration
        current_epsilon = epsilon
        for ep in range(1, num_episodes + 1):
            # build a policy for this episode using current epsilon
            policy_fn_local = lambda s, ce=current_epsilon: epsilon_greedy_action(
                Q, s, actions, ce
            )
            # exploring starts: occasionally begin from a random non-terminal state and random action
            if random.random() < exploring_starts_prob:
                non_terms = [s for s in env.state_space() if not env.is_terminal(s)]
                start_s = random.choice(non_terms)
                start_a = random.choice(actions)
                episode = generate_episode(
                    env, policy_fn_local, start_state=start_s, start_action=start_a
                )
            else:
                episode = generate_episode(env, policy_fn_local)
            # first-visit MC updates (per-(s,a) sample-average update using counts)
            G = 0.0
            visited = set()
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = gamma * G + r
                if (s, a) not in visited:
                    visited.add((s, a))
                    # update count and perform incremental sample-average update:
                    Counts[(s, a)] += 1
                    n = Counts[(s, a)]
                    Q[(s, a)] = Q[(s, a)] + (1.0 / n) * (G - Q[(s, a)])
            # decay exploration rate
            current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

            # update V(start) history
            q_vals_start = [Q[((env.start), a)] for a in actions]
            V_start = float(max(q_vals_start))
            V_start_history.append(V_start)

            # update greedy policy for plotting
            for s in env.state_space():
                if env.is_terminal(s):
                    continue
                q_vals = [Q[(s, a)] for a in actions]
                max_val = max(q_vals)
                max_actions = [a for a, q in zip(actions, q_vals) if q == max_val]
                policy[s] = random.choice(max_actions)

            # update plots every update_interval episodes
            if ep % update_interval == 0 or ep == 1 or ep == num_episodes:
                ax_grid.clear()
                plot_grid_and_policy(ax_grid, env, Q, policy)
                # colorbar (create/destroy to avoid duplicates)
                # (quick hack: imshow again to attach color scale)
                V2 = compute_state_values_from_Q(env, Q)
                im = ax_grid.imshow(V2, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                # annotate current episode and convergence info
                current_v = V_start_history[-1] if V_start_history else 0.0
                ax_grid.text(
                    0.5,
                    -0.12,
                    f"Episode: {ep}/{num_episodes}, V(start): {current_v:.3f}, ε: {current_epsilon:.3f}",
                    transform=ax_grid.transAxes,
                    fontsize=9,
                    va="top",
                )

                plot_value_line(ax_line, V_start_history, window=50)
                plt.pause(0.001)  # allow UI event loop to update

        # final analysis
        final_v = V_start_history[-1] if V_start_history else 0.0
        print(f"\nFinal Results:")
        print(f"  V(start) converged to: {final_v:.4f}")
        print(
            f"  This value reflects: discounted rewards (γ={gamma}), exploration noise,"
        )
        print(f"  and the optimal policy under ε-greedy behavior.")

        # final draw
        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing visualization.")
        plt.ioff()
        plt.show()


# -------------------------
# Entrypoint
# -------------------------


if __name__ == "__main__":
    # Example configurations to demonstrate different behaviors:

    # Configuration 1: Standard setup (will converge around 0.3-0.5)
    print("Running Configuration 1: Standard Monte Carlo")
    run_visual_demo(
        n_rows=5,
        n_cols=5,
        start=(4, 0),
        goal=(0, 4),
        hole=(1, 1),
        num_episodes=2000,
        gamma=0.9,
        epsilon=0.1,
        update_interval=25,
        optimistic_init=0.0,  # Changed from 0.5 to see actual learning
        exploring_starts_prob=0.15,
    )

    # Uncomment below for alternative configurations:

    # Configuration 2: Closer start to goal (higher value)
    # run_visual_demo(
    #     start=(1, 3), goal=(0, 4), hole=(2, 2),
    #     gamma=0.95, optimistic_init=0.0
    # )

    # Configuration 3: Undiscounted (gamma=1.0) for clearer terminal rewards
    # run_visual_demo(
    #     gamma=1.0, epsilon=0.05, optimistic_init=0.0
    # )
