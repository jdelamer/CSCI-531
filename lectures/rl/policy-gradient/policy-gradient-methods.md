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

# Policy Gradient Methods

Until now, we used action-value methods in which the action selected in a state depends of its estimated value.

However, instead of learning an action value function, we will learn a **parameterized policy** that can select actions without a value function.

:::{note}
 While a value function can support policy learning, it is not strictly necessary.
:::


```{margin}
The policy parameter vector $\mathbf{\theta}$ is similar to the weight vector $\mathbf{w}$ introduced previously, and can be implemented with a linear approximation.
```

We define $\mathbf{\theta} \in \mathbb{R}^{d'}$ as the vector of policy parameters. The probability of taking action $a$ in state $s$ at time $t$, given parameters $\mathbf{\theta}$, is defined as:

$$
\pi(a|s,\mathbf{\theta}) = P(a_t = a|s_t = s,\mathbf{\theta}_t = \mathbf{\theta})
$$

The policy parameters are updated using the gradient of a scalar performance measure, $J(\mathbf{\theta})$. Since we aim to maximize this measure, we apply gradient ascent:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_{t} + \alpha\widehat{\nabla J(\mathbf{\theta}_t)}
$$

where $\widehat{\nabla J(\mathbf{\theta}_t)} \in \mathbb{R}^{d'}$ is a stochastic estimate that approximates the gradient of $J$ with respect to $\mathbf{\theta}_t$.

## Policy Approximation and its Advantages

In policy gradient methods, the policy’s parameterization is flexible as long as $\pi(a|s,\mathbf{\theta})$ is differentiable with respect to $\mathbf{\theta}$. In practice we require that the policy never becomes deterministic.

:::{admonition} Activity
:class: activity

Why don’t we want to have a deterministic policy in policy gradient methods?
:::

For discrete action spaces, we can create a parameterized preference $h(s, a, \mathbf{\theta})\in \mathbb{R}$ for each state-action pair. Actions with higher preferences are assigned greater probabilities via a softmax distribution:

$$
\pi(a|s,\mathbf{\theta}) = \frac{e^{h(s,a,\mathbf{\theta})}}{\sum_b e^{h(s,b,\mathbf{\theta})}}
$$

Softmax parameterization offers two advantages:

- The approximate policy can approach a deterministic policy, whereas with $\epsilon$-greedy will always have a probability to choose a suboptimal action. 
- It also enables the selection of actions with arbitrary probabilities. Some problem doesn't have one optimal action for a state, but multiple with different probabilities.


```{code-cell} ipython3
:tags: ["remove-cell"]

from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

h = [2, -1, 0, 1]
det = np.sum([np.exp(h_a) for h_a in h])
pi = []

for h_a in h:
  pi.append(np.exp(h_a)/det)
ax.bar(["a_1","a_2","a_3","a_4"], pi)
glue("ex_softmax", fig, display=False)

```

````{prf:example} Softmax
:label: ex:softmax

To clarify we will take a simple example. We have four actions $A={a_1, a_2, a_3, a_4}$ with the following preferences for the state $s$:
- $h(s,a_1,\theta) = 2$
- $h(s,a_2,\theta) = -1$
- $h(s,a_3,\theta) = 0$
- $h(s,a_4,\theta) = 1$

```{code}
import numpy as np
import matplotlib.pyplot as plt

h = [2, -1, 0, 1]
det = np.sum([np.exp(h_a) for h_a in h])
pi = []

for h_a in h:
  pi.append(np.exp(h_a)/det)
plt.bar(["a_1","a_2","a_3","a_4"], pi)
plt.show()

```

```{glue:figure} ex_softmax
:figwidth: 50%
```

We can see that the action with the highest preference, $a_1$, has more than 60% chance to be selected, while the least prefered action, $a_2$, has a probability of less than 10%, but not 0%!

````


## REINFORCE: Monte Carlo Policy Gradient

As always, we can study a first algorithm based on Monte Carlo as it is always easier to implement.

The policy gradient theorem establishes:

$$
\begin{aligned}
 \nabla J(\mathbf{\theta}) &\propto \sum_s\mu(s)\sum_a q_\pi(s,a)\nabla\pi(a|s,\mathbf{\theta})\\
&= \mathbb{E}_\pi \left[ \sum_a q_\pi(s_t, a)\nabla\pi(a|s_t,\mathbf{\theta}) \right]
\end{aligned}
$$

It provides an analytic expression for the gradient of performance with respect to the policy parameter, but does not involve the derivative of the states distribution.

:::{note}
We will not see the proof, but you can look it up if you're interested.
:::

If we stop there, we could create our gradient-ascent algorithm as:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha\sum_a \hat{q}(s_t, a, \mathbf{w})\nabla\pi(a|s_t,\mathbf{\theta})
$$

where $\hat{q}$ is some learned approximation to $q_\pi$.

However, we want to remove the q-values to only need the policy. So we modify REINFORCE to introduce an action $a$ by replacing the sum over the q-values values by an expectation under $\pi$.

The previous equation does not have the term weighted by $\pi(a|s,\mathbf{\theta})$, so we can change that by multiplying and then dividing the summed terms by $\pi(a|s,\mathbf{\theta})$:

$$
\begin{aligned}
\nabla J(\mathbf{\theta}) &\propto \mathbb{E}_\pi \left[ \sum_a \pi(a|s_t,\mathbf{\theta})q_\pi(s_t,a)\frac{\nabla\pi(a|s_t,\mathbf{\theta})}{\pi(a|s_t,\mathbf{\theta})}) \right]\\
&= \mathbb{E}_\pi\left[q_\pi(s_t,a_t)\frac{\nabla\pi(a_t|s_t,\mathbf{\theta})}{\pi(a_t|s_t,\mathbf{\theta})}) \right],\ \text{($a$ is replaced by $a_t\sim \pi$)}\\
&= \mathbb{E}_\pi\left[G_t\frac{\nabla\pi(a_t|s_t,\mathbf{\theta})}{\pi(a_t|s_t,\mathbf{\theta})}) \right], \text{(because $\mathbb{E}_\pi\left[G_t|a_t,s_t\right] = q_\pi(s_t,a_t)$)}
\end{aligned}
$$

The final expression is a quantity that can be sampled and if we integrate that to our algorithm, we have the following update of the parameters:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha G_t\frac{\nabla\pi (a_t|s_t,\mathbf{\theta}_t)}{\pi (a_t|s_t,\mathbf{\theta}_t)}
$$

Concretly, each increment is proportional to the product of a return $G_t$ and a vector.

The vector is the direction in parameter space that most increases the probability of repeating the action $a_t$ on future visits to state $s_t$.

The update increases the parameter vector in this direction proportional to the return, and inversely proportional to the action probability.

- The former makes sense because it causes the parameter to move most in the directions that favour actions that yield the highest return.
- The latter makes sense because otherwise actions that are frequently selected are at an advantage and might win out even if they do not yield the highest return.


```{prf:algorithm} Monte Carlo policy-gradient
:label: alg:MC-policy

$
\begin{array}{l}
  \textbf{Inputs}:\\
  \quad\quad N\ \text{the number of episodes}\\
  \quad\quad \alpha\in [0, 1]\ \text{the step size}\\
  \quad\quad \text{A policy parameterization}\ \pi (a|s,\theta)\\
  \textbf{Initialize}: \\
  \quad\quad  \theta \in \mathbb{R}^{d'}, \text{e.g. to}\ \mathbf{0}\\
  \textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
  \quad\quad \text{Generate an episode using } \pi: s_0, a_0, r_1, \dots, s_{T-1}, a_{T-1}, r_T\ \text{following}\ \pi (.|.,\theta)\\
  \quad\quad \textbf{Repeat } \text{for each step } t = 0,1 ,\dots, T-1:\\
  \quad\quad\quad\quad G \leftarrow \sum_{k=t+1}^T\gamma^{k-t-1} r_{k}\\
  \quad\quad\quad\quad \theta \leftarrow \theta + \alpha\gamma^t G\nabla\ln\pi(a_t|s_t,\theta)\\
\end{array}
$

```

## Actor-Critic methods

We have seen with TD and SARSA that a one-step update is often more interesting in terms of computation. However, the previous policy-gradient method requires the entire return.

Actor-critic methods allow us to have a one-step policy gradient.

- The idea:

  - We use the q-value to assess the action.
  - When q-values are used this way it is called a critic.
  - And the policy is an actor.

- We replace the full return of REINFORCE by a one-step return as follows:

  $$
  \begin{aligned}
  \mathbf{\theta}_{t+1} &= \mathbf{\theta}_t + \alpha\left( G_{t:t+1}-\hat{v}(s_t,\mathbf{w}_t) \right)\frac{\nabla\pi (a_t|s_t,\mathbf{\theta}_t)}{\pi (a_t|s_t,\mathbf{\theta}_t)}\\
  &= \mathbf{\theta}_t + \alpha\left( r_{t+1} + \gamma\hat{v}(s_{t+1},\mathbf{w})  -\hat{v}(s_t,\mathbf{w}) \right)\frac{\nabla\pi (a_t|s_t,\mathbf{\theta}_t)}{\pi (a_t|s_t,\mathbf{\theta}_t)}\\
  &= \mathbf{\theta}_t + \alpha\delta_t\frac{\nabla\pi (a_t|s_t,\mathbf{\theta}_t)}{\pi (a_t|s_t,\mathbf{\theta}_t)}\\
  \end{aligned}
  $$

- It is combined with the state-value-function of TD(0) and we obtain the following algorithm:

```{prf:algorithm} One-step Actor-critic
:label: alg:AC

$
\begin{array}{l}
  \textbf{Inputs}:\\
  \quad\quad N\ \text{the number of episodes}\\
  \quad\quad \alpha^\theta, \alpha^\mathbf{w}\in [0, 1]\ \text{the step size}\\
  \quad\quad \text{A policy parameterization}\ \pi (a|s,\theta)\\
  \quad\quad \text{A value function parameterization}\  \hat{v}(s,\mathbb{w})\\
  \textbf{Initialize}: \\
  \quad\quad  \theta \in \mathbb{R}^{d'}, \text{e.g. to}\ \mathbf{0}\\
  \quad\quad  \mathbf{w} \in \mathbb{R}^{d}, \text{e.g. to}\ \mathbf{0}\\
  \textbf{Repeat}\ \text{for}\ N\ \text{episodes:}\\
  \quad\quad s \leftarrow Initial state\\
  \quad\quad I \leftarrow 1\\
  \quad\quad \textbf{While } s\ \text{not terminal}:\\
  \quad\quad\quad\quad a \sim \pi (. |s,\theta )\\
  \quad\quad\quad\quad r, s' \leftarrow \text{Execute } a\\
  \quad\quad\quad\quad \delta \leftarrow r + \gamma\hat{v}(s',\mathbf{w}) - \hat{v}(s,\mathbf{w})\\
  \quad\quad\quad\quad \mathbf{w} \leftarrow \mathbf{w} + \alpha^\mathbf{w}\delta\nabla\hat{v}(s,\mathbf{w})\\
  \quad\quad\quad\quad \theta\leftarrow \theta + \alpha^\theta I \delta\nabla\ln \pi(a|s,\theta)\\
  \quad\quad\quad\quad I \leftarrow \gamma I\\
  \quad\quad\quad\quad s \leftarrow s' \\
\end{array}
$

```

## Continuous Actions

- Some problems can have continuous actions.
- Until now the previous methods were suing q-values, thus tabular methods.
- It would be very complicated to adapt this types of methods to continuous actions.
- Thankfully, policy-gradient offers a very practical way to do that!

:::{admonition} Activity
:class: activity

- Give some problems that have very large or continuous action space?
:::

- How does it work:

  - We don't compute the probability for each action.
  - We learn a probability distribution.

- We could choose to use a Gaussian distribution to select the actions:

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

:::{figure} ./gaussians.png
:align: center
:::

- If we want to use a normal distribution with a policy parameterization, then we can rewrite $\pi(a|s,\mathbf{\pi})$ as a normal distribution:

$$
\pi(a|s,\mathbf{\pi}) = \frac{1}{\sigma(s,\mathbf{\theta})\sqrt{2\pi}}\exp\left(-\frac{(a-\mu(s,\mathbf{\theta}))^2}{2\sigma(s,\mathbf{\theta})^2}\right)
$$

- It also implies that the policy's parameter vector is in two parts, $\mathbf{\theta}=[\mathbf{\theta}_\mu, \mathbf{\theta}_\sigma]^T$.
