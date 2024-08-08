# Temporal-Difference Learning

We covered the fundamentals to understand reinforcement learning. One central methods for reinforcement learning is called **Temporal-Difference** (TD) learning. TD learning is composed of dynamic programming and Monte Carlo methods.

## TD Prediction

````{admonition} Activity
:class: activity

Following is the update done by Monte Carlo methods. What is the disadvantage?

```{figure} ./img/MC_update.png
:align: center
```
````

To aoid this issue, TD works differently. TD forms a target at time $t+1$ and we use the reward $r_{t+1}$ and the current estimate $V(s_{t+1})$ to update $V(s_{t})$.

The update is done as:

$$
V(s_t) \leftarrow V(s_t) + \alpha\left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]
$$

:::{figure} ./img/TD_update.png
:align: center
:::

This TD method is called TD(0), because the target update the current step. 

We can write a very simple algorithm to evaluate the policy:

```{prf:algorithm} Algorithm TD(0)
:label: alg:td0

$
\begin{array}{l}
\textbf{Inputs}:\\
\quad\quad \pi\ \text{The policy to be evaluated}\\
\quad\quad N\ \text{the number of episodes}\\
\quad\quad \alpha\in 0, 1\ \text{the step size}\\
\textbf{Initialize}: \\
\quad\quad  V_\pi(s) \in \mathbb{R}, \text{arbitrarily, for all } s \in S,\ \text{except}\ V(terminal)=0 \\
\textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
\quad\quad \textbf{Intialize}\ s\\
\quad\quad \textbf{Repeat}\ \text{for each step until} s=terminal:\\
\quad\quad\quad\quad a\leftarrow \pi(s)\\
\quad\quad\quad\quad r, s' \leftarrow \text{Execute a}\\
\quad\quad\quad\quad V(s) \leftarrow V(s)+\alpha \left[r + \gamma V(s') - V(s)\right]\\
\quad\quad\quad\quad s\leftarrow s'\\
\end{array}
$

```

```{admonition} Activity
:class: activity

What major differences can you see between this algorithm and Monte Carlo?
```

The TD target is an estimate for two reasons:

1. The expected value $V(s')$ is sampled.
2. It uses the current *estimate* $V(s)$.

Moreover $\left[ r + \gamma V(s') - V(s) \right]$ can be seen as an error. It measures the difference between the estimated value $V(s_t)$ and the better estimate $r + \gamma V(s')$.

```{prf:definition} TD error

$$
\delta_t = r_{t+1}+\gamma V(s_{t+1}) - V(s_t)
$$

The TD error depends on the next state and next reward, so it's not available until $t+1$.
```

::::{admonition} Example
:class: example

Some days as you drive to your friend's house, you try to predict how long it will take to get there. When you leave town, you note the time, the day of week, the weather, and anything else that might be relevant. The following table summarizes what happened:

:::{figure} ./img/td-example-table.png
:align: center
:width: 80%
:::

If you apply, Monte Carlo and TD learning you would obtain this:

:::{figure} ./img/td-example-graph.png
:align: center
:width: 80%
:::
::::

```{admonition} What are the advantage of TD learning?

TD methods do not require a model of the environment, the reward function or transition function.
An advantage over Monte Carlo methods is that they are online and fully incremental.
```

## SARSA: On-Policy TD learning

Now that we can estimate the value function of a policy, we can improve it. As previous methods, we also need to trade off exploration or prediction. For on-policy methods we must estimate $q_\pi(s,a)$ for all states $s$ and actions $a$.

:::{admonition} Activity
:class: activity

Why do we need to use the pair state-action?
:::

Concretely, the value function is replaced by the q-values in the update function:

$$
q(s,a) = q(s,a) + \alpha\left[ r_{t+1} + \gamma q(s_{t+1}, a_{t+1}) - q(s_t, a_t) \right]
$$

This update utilizes every element of the transition $(s_,a_t,r_{t+1},s_{t+1},a_{t+1})$.

This algorithm is called **Sarsa** for **s**tate-**a**ction-**r**eward-**s**tate-**a**ction:

```{margin} Action selection
The action selection is important to have a good ratio between exploration and prediction. The strategies seen in previous topics can be used freely.
```

````{prf:algorithm} SARSA
:label: alg:sarsa

$
\begin{array}{l}
\textbf{Inputs}:\\
\quad\quad N\ \text{the number of episodes}\\
\quad\quad \alpha\in 0, 1\ \text{the step size}\\
\textbf{Initialize}: \\
\quad\quad  Q(s, a) \in \mathbb{R}, \text{arbitrarily, for all } s \in S, a\in A,\ \text{except}\ Q(terminal,.)=0 \\
\textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
\quad\quad \textbf{Intialize}\ s\\
\quad\quad a\leftarrow Q(s,.)\ \text{(using }\epsilon\text{-greedy)}\\
\quad\quad \textbf{Repeat}\ \text{for each step until} s=terminal:\\
\quad\quad\quad\quad r, s' \leftarrow \text{Execute a}\\
\quad\quad\quad\quad a'\leftarrow Q(s,.)\ \text{(using }\epsilon\text{-greedy)}\\
\quad\quad\quad\quad Q(s,a) \leftarrow Q(s,a)+\alpha \left[r + \gamma Q(s',a') - Q(s,a)\right]\\
\quad\quad\quad\quad s\leftarrow s'\\
\quad\quad\quad\quad a\leftarrow a'\\
\end{array}
$

````

```{important}
If you look closely, we never mention a policy in the algorithm. The policy can be calculated based on the q-values.
```


## Q-Learning: Off-policy TD learning

The off-policy learning algorithm called **Q-Learning** is old (1989).

The update function is:

$$
q(s_t, a_t) = q(s_t, a_t) + \alpha\left[ r_{t+1} + \gamma\max_{a'} q(s_{t+1}, a')- q(s_t, a_t) \right]
$$

In this case it approximates $q^*$.

````{prf:algorithm} Q-Learning
:label: alg:q-learning

$
\begin{array}{l}
\textbf{Inputs}:\\
\quad\quad N\ \text{the number of episodes}\\
\quad\quad \alpha\in 0, 1\ \text{the step size}\\
\textbf{Initialize}: \\
\quad\quad  Q(s, a) \in \mathbb{R}, \text{arbitrarily, for all } s \in S, a\in A,\ \text{except}\ Q(terminal,.)=0 \\
\textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
\quad\quad \textbf{Intialize}\ s\\
\quad\quad \textbf{Repeat}\ \text{for each step until} s=terminal:\\
\quad\quad\quad\quad a\leftarrow Q(s,.)\ \text{(using }\epsilon\text{-greedy)}\\
\quad\quad\quad\quad r, s' \leftarrow \text{Execute a}\\
\quad\quad\quad\quad Q(s,a) \leftarrow Q(s,a)+\alpha \left[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right]\\
\quad\quad\quad\quad s\leftarrow s'\\
\end{array}
$

````

:::{admonition} Activity
:class: activity

What is the difference between these two algorithms?
:::
