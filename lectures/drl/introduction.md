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

# Introduction to Neural Networks

Neural Networks (NN) are a foundational tool in machine learning, including reinforcement learning. To effectively understand and apply NNs, we begin with an introduction to the necessary mathematical background: **Linear Algebra**.


## Linear Algebra

Linear algebra is deeply linked with neural networks. A solid understanding of its concepts is essential for creating and working with NNs.

### Scalars, Vectors, Matrices, and Tensors

The study of linear algebra involves several types of mathematical objects.

1. **Scalars**

    ```{prf:definition} Scalar
    A scalar is a single number defined on a domain.
    ```

    - **Notation**: Scalars are denoted by italic lowercase letters, e.g., $s \in \mathbb{R}$.

---

2. **Vectors**

    ```{prf:definition} Vector
    A vector is an ordered array of numbers, where each number is identified by its position in the sequence.
    ```

    - **Notation**: Vectors are denoted by bold lowercase letters, e.g., $\mathbf{x}$.
    - **Elements**: Each element is represented by the vector name subscripted with its index, e.g., $x_1, x_2, \ldots$.

    When written explicitly, vectors appear as columns enclosed in square brackets:

    ```{math}
    \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
    ```

---

3. **Matrices**

    ```{prf:definition} Matrix
    A matrix is a two-dimensional array of numbers, where each element is identified by two indices.
    ```

    - **Notation**: Matrices are denoted by bold uppercase letters, e.g., $\mathbf{A}$.
    - **Elements**: Matrix elements are represented by their name in italics with indices, e.g., $A_{m,n}$.

    When written explicitly, matrices are shown as arrays enclosed in square brackets:

    ```{math}
    \mathbf{A} = \begin{bmatrix} A_{1,1} & A_{1,2} \\ A_{2,1} & A_{2,2} \end{bmatrix}
    ```

---

4. **Tensors**

    ```{prf:definition} Tensor
    A tensor is an $n$-dimensional array of numbers, where each element is identified by $n$ indices.
    ```

    - **Notation**: Tensors are generalized multi-dimensional arrays, often denoted with boldface names.
    - **Elements**: Tensor elements are represented by their name in italics with indices, e.g., $A_{i,j,k}$.


### Operations

With these mathematical object, we can do different operations.

#### Basic Operations

One important operation on matrices is the **transpose**.

The transpose of a matrix is the mirror image across the **main diagonal**. We denote the transpose of matrix $\mathbf{A}$, as $\mathbf{A}^\intercal$, such that $\left(\mathbf{A}^\intercal\right)_{i,j} = A_{j,i}$.


````{prf:example} Transpose example

```{math}

\mathbf{A} = \begin{bmatrix}
A_{1,1} & A_{1,2} \\ A_{2,1} & A_{2,2} \\ A_{3,1} & A_{3,2}
\end{bmatrix} \Rightarrow \mathbf{A}^\intercal = \begin{bmatrix}
A_{1,1} & A_{2,1} & A_{3,1} \\ A_{1,2} & A_{2,2} & A_{3,2}
\end{bmatrix}
```

````

It can be verified with a small python code.

```{code-cell} ipython3
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(np.transpose(A))
```

---

We can also add matrices to each other, as long as they have the same shape. Consider two matrices $\mathbf{A} \in \mathbb{R}^{m\times n}$ and $\mathbf{B} \in \mathbb{R}^{m\times n}$, then $\mathbf{C} = \mathbf{A} + \mathbf{B}$, where $C_{i,j} = A_{i,j} + B_{i,j}$.

```{code-cell} ipython3
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

B = np.array([[10, 11],
              [12, 13],
              [14, 15]])
print(A+B)
```

We can also add or multiply a scalar to matrix, such as $\mathbf{D} = a \cdot \mathbf{B} + c$, where $D_{i,j} = a \cdot B_{i,j} + c$.

Similarly, we can verify ot by running some python code.

```{code-cell} ipython3
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print(2*A-1)
```

#### Multiplying Matrices and Vectors

The **matrix product** of matrices $\mathbf{A}$ and $\mathbf{B}$ is a third matrix $\mathbf{C}$. To be defined $\mathbf{A}$ must have the same number of columns as $\mathbf{B}$ has rows. Concretely if $\mathbf{A}\in \mathbb{R}^{m\times n}$, and $\mathbf{B}\in \mathbb{R}^{n\times p}$, then $\mathbf{C}\in \mathbb{R}^{m\times p}$.

We can matrix product by placing two matricres together, for example: $\mathbf{C} = \mathbf{A}\mathbf{B}$.

The product operation is defined by:

```{math}
C_{i,j} = \sum_{k} A_{i,k}B_{k,j}
```

In the following code we have $A\in \mathbb{R}^{3\times 2}$, and $B\in \mathbb{R}^{2\times 4}$, so the result matrix will be $C\in \mathbb{R}^{3\times 4}$.

```{code-cell} ipython3
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

B = np.array([[2, 3, 1, 4], [2, 2, 1, 5]])

print(np.matmul(A,B))
```

Not to mistake it with the element-wise product denoted $\mathbf{A}\odot \mathbf{B}$.

```{code-cell} ipython3
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

B = np.array([[2, 3],
              [1, 2],
              [2, 1]])

print(A*B)
```

---

It is now enough information to write a system of linear equations:

```{math}
\mathbf{Ax} = \mathbf{b}
```

where $\mathbf{A}\in \mathbb{R}^{m\times n}$ is known matrix, $\mathbf{b}\in \mathbb{R}^{m}$ is known vector, and $\mathbf{x}\in \mathbb{R}^{n}$ is a vector of unknown variables.

Each element $x_i$ of $\mathbf{x}$ is an unknown variable, and each row of $\mathbf{A}$ and each element of $\mathbf{b} provide another constraint.

We can rewrite the previous equation as:

```{math}
\begin{align}
\mathbf{A}_{1,:}\mathbf{x} &= b_1\\
&\dots \\
\mathbf{A}_{m,:}\mathbf{x} &= b_m
\end{align}
```
or even more explicitly:

```{math}
\begin{align*}
A_{1,1}x_1+ A_{1,2}x_2 + \dots &+ A_{1,n}x_n = b_1\\
\dots &  & \\
A_{m,1}x_1+ A_{m,2}x_2 + \dots &+ A_{m,n}x_n = b_m
\end{align*}
```

```{note}
Matrix-vector product notation provides a more compact representation for equations of this form.
```

#### Solving Linear Equations

We have seen linear algebra can be used to represent a system of equations, but it can also be used to solve them.

````{prf:definition} Identity Matrix
An identity matrix $\mathbf{I}$ is a matrix that doesn't change any vector, when multiplying the vector by that matrix. Formally, $\mathbf{I} \in \mathbf{R}^{n\times n}$ and,

```{math}
\forall \mathbf{x} \in \mathbb{R}^n,\ \mathbf{I}_n\mathbf{x} = \mathbf{x}
```
````

```{prf:example} Idendity matrix for $n=3$
For example $\mathbf{I_3}$ is $\begin{bmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\end{bmatrix}$.
```


Using the identity matrix we can define another tool called **matrix inversion**.

````{prf:definition} Matrix Inverse
The matrix inverse of $\mathbf{A}$ is denoted $\mathbf{A}^{-1}$, and defned such as:
```{math}
\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}_n
```
````

It is now possible to solve our system of equations following these steps:

```{math}
\begin{align}
\mathbf{Ax} &= \mathbf{b}\\
\mathbf{A}^{-1}\mathbf{Ax} &= \mathbf{A}^{-1}\mathbf{b}\\
\mathbf{I}_n\mathbf{x} &= \mathbf{A}^{-1}\mathbf{b}\\
\mathbf{x} &= \mathbf{A}^{-1}\mathbf{b}
\end{align}
```

```{code-cell} ipython3
import numpy as np
from numpy.linalg import inv

A = np.array([[1, 1, 1],
              [0, 2, 5],
              [2, 5, -1]])

b = np.array([6, -4, 27])
ainv = inv(A)

x = np.matmul(ainv, b)
print(x)
```

```{important}
For $\mathbf{A}^{-1}$ to exist, the equation must have **exactly** one solution for every value of $\mathbf{b}$.
```

## Deep Neural Networks

In previous topic, we saw tht we could calculate an approximate value function $\hat{v}(s,\mathbf{w})$. The weight vector $\mathbf{w}$ is learned by using a gradient descent methods.

This approach represent $\hat{v}(s,\mathbf{w})$ by a linear function with respect to the feature vector $\mathbf{x}(s)$, and finding a states features can be non-trivial.

In contrast to linear function approximation, deep learning provides a universal method for function approximation that is able to automatically learn feature representations of states, and is able to represent non-linear, complex functions that can generalize to novel states.

### Feedforward Neural Networks

Feed forward networks (FFN) are the most common type of networks in RL. FFN are build in *layers* with the first layer processing the input $\mathbf{x}$ and any subsequent layer processing the output of the previous layer.

Each layer is defined as a parameterized function, and is composed of many units. The output is the concatenation of each unit output.

Usually we refer to some layers with spcific terms:

- Input layer: first layer of the network.
- Output layer: Last layer of the network.
- Hidden layers: All layers between the input and output layers.

````{prf:example} FFN example

Consider a neural network with three layers:

```{image} ./img/ffn_example.png
    :align: center
```

This FFN could be written as:

```{math}
f(\mathbf{x}, \mathbf{\theta}) = f_3(f_2(f_1(\mathbf{x}, \mathbf{\theta}^1), \mathbf{\theta}^2), \mathbf{\theta}^3)
```

where $\theta^k$ are the parameters of layer $k$ and $\theta = \bigcup_k \theta^k$.
````

#### Neural Unit

A neural unit of a layer $k$ represents a parameterized function $f_{k,u}: \mathbb{R}^{d_{k-1}} \rightarrow \mathbb{R}$, where $d_{k}$ is the number of units at the layer depth $k$.

Each unit computes a linear transformation followed by a non-linear **activation function** $g_k$.

The linear transformation is computed following:

```{math}
y = \mathbf{x}\mathbf{w}^\intercal + b
```

where $\mathbf{w} \in \mathbb{R}^{d_{k-1}}$ is a weight vector, and $b\in \mathbb{R}$ is a bias.

```{important}
The parameters $\theta^k_u\in \mathbb{R}^{d_{k-1}+1}$ contains the weight vector $\mathbf{w}$ and the bias $b$.
```
So the unit computation can be written as:

```{math}
f_{k,u}(\mathbf{x},\theta^{k}_u) = g_k(\mathbf{x}\mathbf{w}^\intercal + b)
```

```{important}
Activation function are crucial.

The composition of linear functions only result in a linear function.

Addign a non-linear activation function between layers allow the network to learn complex non-linear aproximations.
```

````{prf:example} Neural Unit example
Consider a neural unit receiving a vector $\mathbf{x} \in \mathbb{R}^3$.

```{image} ./img/unit_example.png
    :align: center
    :width: 60%
```
````

#### Activation Functions

There are many activation functions, some are more common than overs.

In RL, ReLU or rectified linear unit applies a non-linear transformation bu remains "close to linear". This has useful implications for gradient-based optimization. Also ReLU is able to output zero values.

The tanh and sigmoid activation functions are mostly used to restrict the output of a neural network to be within the ranges of $(-1,1)$ or $(0,1)$, respectively.


```{image} ./img/activation_functions.png
    :align: center
```
