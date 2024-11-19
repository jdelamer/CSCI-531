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


````{admonition} Example
:class: example

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
