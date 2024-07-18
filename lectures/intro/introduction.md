# Introduction


## What is Reinforcement Learning (RL)?

* Reinforcement learning is what to do to maximize a reward.
* We can give a more "formal" definition.

````{prf:definition} Reinforcement Learning
:label: def-rl

Reinforcement Learning is calculating a function that maps situations to actions.
````

*  We said that we want to maximize a reward, but what is a reward?

```{admonition} Activity
:class: activity
Try to explain what a reward is.
```

*  To maximize a reward the learner can do different actions.
*  If the learner was passive, it could not maximize anything.
*  Usually, the learner start with no prior knowledge about what action it should do.

```{admonition} Activity
:class: activity

What would you do to maximize a reward if you had no idea which action you should do?
```

Key concepts
------------

* Reinforcement learnign has two main characteristics:
  * It use a *trial-and-error* type of search.
  * It has delayed rewards.

```{warning}

Reinforcement learning is a name that regroups different concepts:
* It's a type of problem.
* It's also a class of solution methods.
* And it's the field that study the two previous points as well.

You need to understand the distinction.
```

## What is the Reinforcement learning problem? (Simplified)

* The reinforcement learning problem is an idea coming from dynamical system theory.
* And more specificaly from the Markov Decision Processes.
* The basic ideas are:
  * A learning agent must *sense* the state of the environment.
  * The agent must be able to take *actions* that affect the state.
  * It must have a *goal* or goals relating to the state of the environment.

```{image} ./img/rl.drawio.png
    :align: center
```

## What is a Reinforcement learning method?


*  It is any methods that is well suited for solving RL problems.

```{admonition} Activity
:class: activity

*  Discuss why chess can be considered a reinforcement learning problem.
*  Try to define for chess the states, actions, and goal.
*  Why the goal must be related to the state?
```

## What Reinforcement learning is not?

* Supervised learning
  * Supervised learning is from a training set of *labeled* examples.
  * Each example describes a situation and the action to take in this situation.
  * The objective is to extrapolate/generalize from this set of examples.
  * However, in interactive problem it is impractical to obtain examples that are both correct and representative.
* Unsupervised learning
  * Finding structure hidden in *unlabeled* examples.
  * You could think that is the same, because it doesn't rely on examples of correct behaviors.

```{admonition} Activity
:class: activity

What is the difference?
```

## The challenges of reinforcement learning.

Reinforcement learning has a lot of challenges:

* Trade-off between exploration and exploitation
  * To obtain a lot of reward the agent must choose the action that it tried and found effective.
  * However, to discover such actions it needs to try previously unselected actions.
  * In the first case, we say that the agent exploit.
  * In the second we say that the agent explore.
  * You can neither do one or the other only, you need both.
* Considering the whole problem:
  * Reinforcement learning consider the whole problem.
  * Start with a complete interactive, goal seeking agent.
  * Assumed that the agent has to operate despite uncertainty about the environment.

# Elements of reinforcement learning

```{admonition} Activity
:class: activity

What are the two elements we talked about that compose reinforcement learning?
```

* Reinforcement learning is also composed of:
  * A policy
  * A reward function
  * A value function
  * A model of the environment (optional)

````{prf:definition} Policy
:label: policy_basic

A policy is a function that maps each state to an action.
````

* It define the behavior of an agent at a given time.
* It is the core of reinforcement learning agent, because it is sufficient to determine its behavior.
* It can either be:
  * Deterministic
  * Stochastic

```{image} ./img/policy.drawio.png
:align: center
```

````{prf:definition} Reward
:label: reward

A reward is a value returned by the environment at a time step $t$.
````

*  It defines the goal of a reinforcement learning problem.
*  Remember that the agent's objective is to maximize the total reward it receives.

````{prf:definition} Value Function
:label: value_function_basic

A value function is a function returning for each state the total expected reward starting from this state.
````

```{admonition} Activity
:class: activity

* What is the difference between the reward and the value function?
* Between the following state which one would you select?
  * $s_1$: reward 100, value function 3
  * $s_2$: reward 1, value function 5

```

* We seek actions that lead us to states that bring higher value not higher reward.
* Unfortunately, it is harder to determine the value than the reward.

````{prf:definition} Model
:label: model

The model of the environment is the representation of the dynamic of the problem.
````

*  For a given state and action, the model is able to predict the next state and next reward.
*  We consider two types of reinforcement learning methods:
  *  Model-based
  *  Model-free

## Limitations


*  Reinforcement learning relies heavily on the concept of state.
*  It is the input of the policy and the value function.
*  We will not discuss how we design or model the state.
*  However, it has a great impact on the efficiency of the method.
