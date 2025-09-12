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

# From MDP Definition to Solution: Why We Need More

Now that we understand what an MDP is—a mathematical framework with states, actions, transitions, and rewards—you might be wondering: **"How do we actually solve it?"**

This is where many students feel a conceptual gap. We've defined the problem, but what does it mean to "solve" an MDP? And why do we suddenly need concepts like policies and value functions?

## The Central Question: What Should the Agent Do?

Let's revisit our MDP framework:

:::{figure} ./img/mdp.drawio.png
:align: center
:width: 50%
:::

The MDP tells us the rules of the world, but not how to play the game optimally.

:::{admonition} Example: Autonomous Lab Robot
:class: example

Consider a lab robot with a perfectly defined MDP:
- **States**: All possible (x,y,orientation) positions in the lab
- **Actions**: {move_forward, turn_left, turn_right, stop}
- **Transitions**: Known movement dynamics with noise
- **Rewards**: +100 for reaching target, -1 per step, -50 for hitting equipment

**The Question**: Given this MDP, what should the robot do in each state?
:::

## Three Fundamental Challenges

### 1. The Decision Problem
In any given state $s$, the agent faces a choice among multiple actions. How should it choose?

Naive approaches like random selection ignore available information, while greedy immediate reward maximization ($\arg\max_a r(s,a)$) ignores future consequences. Fixed rules like "always go right" don't adapt to different states.

We need a systematic way to map states to actions that considers long-term consequences.

### 2. The Evaluation Problem
How do we measure if our decision-making strategy is good?

:::{admonition} Think About It
:class: note

If two different strategies both eventually reach the goal, how do we compare them?
- Strategy A: Reaches goal in 10 steps on average
- Strategy B: Reaches goal in 15 steps but avoids risky areas

Which is better? We need a way to evaluate and compare strategies.
:::

### 3. The Optimization Problem
Among all possible strategies, which one is best?

This isn't just about finding *a* solution—we want the *optimal* solution that maximizes long-term reward.

## Introducing Our Solution Framework

To solve these three challenges, we need two key concepts:

A **policy** ($\pi$) serves as our decision maker—a strategy that tells the agent what action to take in each state. This solves the decision problem systematically by providing a mapping from states to actions (or action probabilities).

A **value function** ($v_\pi$) acts as our evaluator—a function that estimates how good it is to be in each state under a given policy. This solves the evaluation problem by letting us compare different policies. Formally, it represents the expected long-term reward starting from each state.

## The Solution Strategy: A Preview

Here's how these concepts work together to solve MDPs:

```
MDP Definition (S, A, T, R)
            ↓
    Choose/Improve Policy π
            ↓
    Evaluate Policy π (Calculate v_π)
            ↓
        Is π optimal?
        ↙        ↘
      No          Yes
       ↓           ↓
   (loop back)   Optimal Policy π*
                 Problem Solved!
```

This iterative process follows a simple pattern:
1. Start with some policy (even random)
2. Evaluate how good this policy is using value functions
3. Improve the policy based on this evaluation
4. Repeat until we find the optimal policy

## Why This Approach Works: Mathematical Guarantees

This framework comes with theoretical guarantees. For every finite MDP, an optimal policy exists. The optimal value function is unique (though multiple optimal policies may exist), and our iterative algorithms will find the optimal solution.

## Connection to Research Applications

This framework scales to real research problems. In drug discovery, the policy determines which molecular modifications to try next, while the value function estimates the expected probability of finding effective compounds from the current molecular state. For robot learning, the policy provides motor control commands based on sensor inputs, and the value function estimates expected task completion probability from the current robot configuration. In LLM training, the policy adjusts training hyperparameters based on the current model state, while the value function estimates expected final model performance.

## What's Coming Next

In the following sections, we'll formalize these intuitive concepts:

1. **Policies**: Mathematical definition and types of policies
2. **Value Functions**: How to calculate and interpret them
3. **Bellman Equations**: The recursive structure that makes everything computable
4. **Optimization Algorithms**: Policy iteration and value iteration

Each concept builds on this foundation, so keep this "big picture" in mind as we dive into the mathematical details.

:::{admonition} Key Insight
:class: note

The MDP framework's power lies in its separation of concerns: MDP definition captures the physics of the problem, policies capture the decision-making strategy, value functions capture the quality assessment, and algorithms handle the optimization. This separation makes complex problems tractable and allows for principled algorithm design.
:::