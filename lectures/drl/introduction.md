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


