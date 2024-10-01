# Monte Carlo Methods

Dynamic programming is a method to calculate the optimal policies of a MDP. One issue with is that it requires complete knowledge about the environment, but in some problems we don't assume complete knowledge and we call them **Model-Free** problems. To calculate a policy without this knowledge, the agent can only use its **experience**. The experience comes from the interaction of the agent with the environment.

:::{figure} ./img/trajectory.drawio.png
:align: center
:::

:::{important}
Although a model is still necessary for interaction, a full transition function is not required to perform Monte Carlo methods.
:::

## Monte Carlo Prediction

The first step in understanding Monte Carlo methods is to evaluate the value of a given policy, denoted as $v_\pi(s)$. 

### Key Idea
- The idea is to use **episodes** (sequences of states and actions from start to a terminal state) to estimate the value function.
- Every time a state $s$ appears in an episode, it's called a **visit**.
- $v_\pi(s)$ is calculated as the average return (total accumulated reward) of all visits to state $s$ across multiple episodes.

```{important}
The Monte Carlo prediction algorithm is guaranteed to converge to the true value $v_\pi(s)$ as the number of visits to state $s$ approaches infinity.
```

In practice, a finite number of episodes, $N$, is used to approximate $v_\pi(s)$. Additionally, instead of averaging returns for every visit to a state, a common approach is to only average the returns from the **first visit** of each state in an episode. This is called **First-Visit Monte Carlo**; alternatively, considering every visit is known as **Every-Visit Monte Carlo**.

```{margin} Choosing $N$
Choosing $N$ can be challenging, it will depend on the problem. The main issue will be the size of the state space and the action space.
$N$ needs to be large enough, the algorithm can explore as many different trajectories as possible, but not too large so it can be computed in a reasonable time. 
```

```{prf:algorithm} Monte Carlo Prediction
:label: alg:MC-prediction

$\begin{array}{l}
  \textbf{Inputs}:\ \text{A policy } \pi \text{ to be evaluated}, N\ \text{the number of episodes}\\
  \textbf{Output}:\ \text{The value function} V_\pi\\
  \textbf{Initialize}: \\
  \quad\quad  V_\pi(s) \in \mathbb{R}, \text{arbitrarily, for all } s \in S \\
  \quad\quad Returns(s) \leftarrow \text{an empty list, for all } s \in S\\ 
  \textbf{Repeat } \text{for } N \text{ episodes:}\\
  \quad\quad \text{Generate an episode using } \pi: S_0, A_0, R_1, S_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T\\
  \quad\quad G \leftarrow 0\\
  \quad\quad \textbf{Repeat } \text{for each step } t = T-1, T-2, \dots, 0:\\
  \quad\quad\quad\quad G \leftarrow \gamma G + R_{t+1}\\
  \quad\quad\quad\quad \textbf{if } S_t \notin S_0, S_1, \dots, S_{t-1}:\\
  \quad\quad\quad\quad\quad\quad \text{append } G \text{ to } Returns(S_t) \\
  \quad\quad\quad\quad\quad\quad  V_\pi(s_t) \leftarrow avg(Returns(S_t))
\end{array}
$

```

::::{admonition} Activity
:class: activity

- Consider the following episode.

:::{figure} ./img/episode.drawio.png
:align: center
:::

- In this episode, one state is visited twice.
- Calculate the return $G$ for this state for both the first and every visit.
::::

## Mont Carlo Estimation of Action Values

:::{admonition} Activity
:class: activity

How can you apply Monte Carlo methods when you don't have a model?
:::

It changes the way the policy is estimated. In this case we apply this method to a state-action pair because, we want to estimate $q^*(s,a)$. Thus, we will average the returns for each pair state-action.

:::{admonition} Activity
:class: activity

Discuss what problems occur with this method.
:::

Due to this problem, all the actions need to be visited to obtain a good estimate. It is similar to the multi-armed bandit problem in which we were trying to balance exploration and exploitation.

:::{note}
Techniques from multi-armed bandit problems, such as $\epsilon$-greedy strategies, can be applied to balance exploration and exploitation.
:::

## Monte Carlo Learning

Monte Carlo methods not only estimate the value function but also allow for **policy improvement**. Once we evaluate a policy, we can improve it iteratively.

### On-Policy vs. Off-Policy
There are two primary approaches to learning with Monte Carlo:

1. **On-policy methods**: The agent learns the value of the policy it is currently following.
2. **Off-policy methods**: The agent learns the value of a different policy (called the target policy) while following a separate behavior policy for exploration.

### On-policy Method

An on-policy agent uses a **soft policy** (a policy that has non-zero probabilities for all actions) and gradually shifts toward a deterministic, optimal policy.

:::{figure} ./img/on-policy_policy.png
:align: center
:scale: 35%
:::

There are many types of soft policies for On-Policy Monte Carlo methods, but the one we will see is the $\epsilon$-greedy policy.
- Nongreedy actions are given minimal probability of selection $\frac{\epsilon}{|A|}$.
- The greedy action gets the remaining probability $1 - \epsilon+\frac{\epsilon}{|A|}$.

````{prf:algorithm} $\epsilon$-greedy policy

$\begin{array}{l}
  \textbf{Inputs}:\\ 
  \quad\quad \text{Small}\ \epsilon > 0 \\
  \quad\quad N\ \text{number of episodes} \\
  \textbf{Output}:\ \text{A policy} \pi\\
  \textbf{Initialize}: \\
  \quad\quad \pi \rightarrow \text{an arbitrary}\ \epsilon\text{-soft policy}\\
  \quad\quad Q(s,a) \in \mathbb{R}\ \text{(arbitrary), for all}, s\in S, a\in A\\
  \quad\quad Returns(s,a) \leftarrow \text{an empty list, for all } s \in S, a\in A\\
  \textbf{Repeat } \text{for } N \text{ episodes:}\\
  \quad\quad \text{Generate an episode using } \pi: S_0, A_0, R_1, S_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T\\
  \quad\quad G \leftarrow 0\\
  \quad\quad \textbf{Repeat } \text{for each step } t = T-1, T-2, \dots, 0:\\
  \quad\quad\quad\quad G \leftarrow \gamma G + R_{t+1}\\
  \quad\quad\quad\quad \textbf{if } S_t \notin S_0, S_1, \dots, S_{t-1}:\\
  \quad\quad\quad\quad\quad\quad \text{append } G \text{ to } Returns(S_t, A_t) \\
  \quad\quad\quad\quad\quad\quad Q(S_t,A_t) \leftarrow avg(Returns(S_t, A_t))\\
  \quad\quad\quad\quad\quad\quad A^* \leftarrow \arg\max_{a} Q(S_t,a)\\
  \quad\quad\quad\quad\quad\quad \textbf{for all}\ a \in A:\\
  \quad\quad\quad\quad\quad\quad\quad\quad \pi (a|S_t) \leftarrow \begin{cases} 1-\epsilon + \epsilon /|A| & \text{if}\ a=A^*\\  \epsilon /|A| & \text{if}\ a\neq A^*\end{cases}\\
\end{array}
$
````

This algorithm guarantee that for any $\epsilon$-soft policy $\pi$, any $\epsilon$-greedy with respect to $q_\pi$ is guaranteed to be better than or equal to $\pi$.

:::{prf:proof}

It is assured by the policy improvement theorem:

$$
\begin{aligned}
q_\pi(s, \pi'(s)) &= \sum_{a}\pi'(a|s)q_\pi(s,a)\\
&=\frac{\epsilon}{A}\sum_{a}q_\pi(s,a)+(1-\epsilon)\max_aq_\pi(s,a)\\
&\geq \frac{\epsilon}{|A|}\sum_{a}q_\pi(s,a)+(1-\epsilon)\frac{\pi(a|s)-\frac{\epsilon}{|A|}}{1-\epsilon}q_\pi(s,a)\\
&=\frac{\epsilon}{|A|}\sum_{a}q_\pi(s,a)-\frac{\epsilon}{|A|}\sum_{a}q_\pi(s,a)+\sum_a\pi(a|s)q_\pi(s,a)\\
&= v_\pi(s)
\end{aligned}
$$

Thus, by the policy improvement theorem $\pi'\geq\pi$.
:::

```{note} 
It achieves the best policy among the $\epsilon$-soft policies.
```

### Off-policy methods

On-policy methods learn action values for a near-optimal policy, because the exploratory part of the method always generate a small number of less optimal actions.

Another approach is to use two policies, one that is learned and that becomes the optimal policy called the **target policy**, and the other that contains the exploratory component called the **behavior policy**.

:::{figure} ./img/off-policy_policies.png
:align: center
:scale: 35%
:::

Using two different policies requires to redefine how we estimate the value function of the target policy and how we can improve it.

#### Prediction problem

Let's simplify the issue for a moment and consider the following problem. 

The target and behavior policies are fixed and we just try to estimate $v_\pi$. We don't have episodes generated by $\pi$; only episodes generated by the behavior policy $b$ with $\pi\neq b$. The only option is to use the episode generated by $b$ to estimate $\pi$.

```{prf:assumption} Coverage
:label: assumption:coverage

Every action taken under $\pi$ is also taken under $b$, meaning, $\pi(a|s)>0$ implies $b(a|s)>0$.
```

```{admonition} Activity
:class: activity

Why is this assumption important?
```

In Off-policy methods, the action values are estimated using **importance sampling**. Importance sampling takes the returns of the trajectories, and weights them relatively to their probability of occurring in both policies. It is called the *importance-sampling ratio*.

Concretely, giving a starting state $s_t$; the probability of the subsequent state-action trajectory $a_t,s_{t+1},a_{t+1},\dots,s_T$ occurring under $\pi$ is:

$$
\begin{aligned}
&P(a_t,s_{t+1},a_{t+1},\dots,s_T|s_t,a_{t:T-1}\sim \pi)\\
&= \prod_{k=t}^{T-1}\pi(a_k|s_k)p(s_{k+1}|s_k,a_k)
\end{aligned}
$$

The following figure illustrate how the probability can be calculated:

```{figure} importance_sampling.drawio.png
:align: center
:scale: 30%
```


```{prf:definition} Importance-sampling Ratio

$$
\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1}\pi(a_k|s_k)p(s_{k+1}|s_k,a_k)}{\prod_{k=t}^{T-1}b(a_k|s_k)p(s_{k+1}|s_k,a_k)} = \prod_{k=t}^{T-1}\frac{\pi(a_k|s_k)}{b(a_k|s_k)}
$$

```

Again, it can be illustrated by the following figure:

```{figure} ./img/importance-ratio.png
:align: center
:width: 85%
```

:::{important}
The importance sampling ratio ends up depending only on the two policies and the sequence not on the MDP.
:::

We wish to estimate the expected returns under the target policy $\pi$.

- We have the returns $G_t$ under $b$.
- The expectation $\mathbb{E}[G_t|s_t=s]=v_b(s)$ cannot be averaged to obtain $v_\pi$.
- We use the ratio $\rho_{t:T-1}$ to transform the returns to have the right expected value:

$$
\mathbb{E}[\rho_{t:T-1}G|s_t=s]=v_\pi(s)
$$

Concretely:

- We define $\mathcal{T}(s)$, the set of all times steps in which state $s$ is visited.
- We use it to obtain the value function update:

$$
V(s) = \frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}G_t}{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}}
$$

:::{note}
It considers that we have the entire trajectories to update it.
:::

We can modify it to have an incremental implementation. Suppose we have a sequence of returns $G_1, G_2, \dots, G_{n-1}$, all starting in the same state, and with corresponding random weight $W_i = \rho_{t_i:T(t_i)-1}$.

We want can rewrite the previous equation:

$$
V_n = \frac{\sum_{k=1}^{n-1}W_kG_k}{\sum_{k=1}^{n-1}W_k}, n\geq 2
$$


To obtain an incremental implementation we need to maintain for each state the cumulative sum $C_n$ of the weights given the first $n$ returns.

The update rule for $V_n$ is:

$$
V_{n+1} = V_n + \frac{W_n}{C_n}\left[ G_n - V_n \right]
$$

with

$$
C_{n+1} = C_n + W_{n+1}
$$

:::{important}
We saw this in the multi-armed bandit!
:::

We can now write an algorithm:

```{prf:algorithm} Off policy prediction
:label: alg:MC-off-pred

$
\begin{array}{l}
  \textbf{Inputs}:\\
  \quad\quad \text{A target policy } \pi \text{ to be evaluated}\\
  \quad\quad N\ \text{the number of episodes}\\
  \textbf{Output}:\ \text{The value function} V_\pi\\
  \textbf{Initialize}: \\
  \quad\quad  Q(s,a) \in \mathbb{R}, \text{arbitrarily, for all } s \in S, \text{for all}\ a \in A\\
  \quad\quad C(s, a) \leftarrow 0\\
  \textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
  \quad\quad b \leftarrow \text{any policy with coverage of}\ \pi\\
  \quad\quad \text{Generate an episode using } \pi: S_0, A_0, R_1, S_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T\\
  \quad\quad G \leftarrow 0\\
  \quad\quad W \leftarrow 1\\
  \quad\quad \textbf{Repeat } \text{for each step } t = T-1, T-2, \dots, 0\ \textbf{and}\ W\neq 0:\\
  \quad\quad\quad\quad G \leftarrow \gamma G + R_{t+1}\\
  \quad\quad\quad\quad C(S_t, A_t) \leftarrow C(S_t,A_t) + W\\
  \quad\quad\quad\quad Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \frac{W}{C(S_t,A_t)}\left[G - Q(S_t,A_t)\right]\\
  \quad\quad\quad\quad W \leftarrow W + \frac{\pi (A_t|S_t)}{b(A_t|S_t)}\\
\end{array}
$
```

#### Off-policy control

To calculate the optimal policy using an off-policy method, we consider the policy $\pi$ to be a greedy policy. After each update of the value function, we assign the best action to the policy.


```{prf:algorithm} Off policy control
:label: alg:MC-off-cont

$
\begin{array}{l}
  \textbf{Inputs}:\\
  \quad\quad N\ \text{the number of episodes}\\
  \textbf{Output}:\ \text{A policy}\ \pi\\
  \textbf{Initialize}: \\
  \quad\quad  Q(s,a) \in \mathbb{R}, \text{arbitrarily, for all } s \in S, \text{for all}\ a \in A\\
  \quad\quad C(s, a) \leftarrow 0\\
  \quad\quad \pi(s) \leftarrow \arg\max_a Q(s,a)\\
  \textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
  \quad\quad b \leftarrow \text{any policy with coverage of}\ \pi\\
  \quad\quad \text{Generate an episode using } \pi: S_0, A_0, R_1, S_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T\\
  \quad\quad G \leftarrow 0\\
  \quad\quad W \leftarrow 1\\
  \quad\quad \textbf{Repeat } \text{for each step } t = T-1, T-2, \dots, 0\ \textbf{and}\ W\neq 0:\\
  \quad\quad\quad\quad G \leftarrow \gamma G + R_{t+1}\\
  \quad\quad\quad\quad C(S_t, A_t) \leftarrow C(S_t,A_t) + W\\
  \quad\quad\quad\quad Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \frac{W}{C(S_t,A_t)}\left[G - Q(S_t,A_t)\right]\\
  \quad\quad\quad\quad \pi(S_t) \leftarrow \arg\max_a Q(S_t,a)\\
  \quad\quad\quad\quad \textbf{If} A_t \neq \pi(S_t)\ \textbf{Then}\text{ exit inner loop (procceed to next episode)}\\
  \quad\quad\quad\quad W \leftarrow W + \frac{\pi (A_t|S_t)}{b(A_t|S_t)}\\
\end{array}
$
```
## Summary

Here is a comparison of On-Policy and Off-Policy Monte Carlo methods:

| **On-Policy**        | **Off-Policy**                                    |
|:---------------------|--------------------------------------------------:|
| Generally simpler    | Requires additional concepts such as importance sampling |
| Policy is evaluated and improved simultaneously | Uses data generated by a different policy (behavior policy) |
| Converges faster but explores less broadly | Greater variance and slower convergence due to importance sampling |
| Less powerful but easier to implement | More powerful and can use any exploratory behavior |

By understanding and applying these Monte Carlo techniques, agents can learn optimal policies in model-free environments through experience, enabling them to adapt to dynamic and complex scenarios.