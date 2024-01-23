---
title: Learning Operators
alias:
  name: operators
  text: Learning Operators
---

# Introduction

Function operators are ubiquitous in mathematics and physics: They are used to
describe dynamics of physical systems, such as the Navier-Stokes equations in
fluid dynamics. As solutions of these systems are functions, it is natural to
transfer the concept of function mapping into machine learning.

## Operators

In mathematics, _operators_ are function mappings – they map functions to functions.

Let $u: X \subset \mathbb{R}^d \to \mathbb{R}^c$ be a function that maps a
$d$-dimensional input to $c$ output *channels*.

An **operator**
$$
G: u \to v
$$
maps $u$ to a function $v: Y \subset \mathbb{R}^{p} \to \mathbb{R}^{q}$.

!!! example annotate
    The operator $G: u \to \partial_x u$ maps functions $u$ to their
    partial derivative $\partial_x u$.

## Learning Operators

Learning operators is the task of learning the mapping $G$ from data.
In the context of neural networks, we want to learn a neural network $G_\theta$
with parameters $\theta$ that, given a set of input-output pairs $(u_k, v_k)$,
maps $u_k$ to $v_k$. We refer to such a neural network as **neural operator**.

In **Continuity**, we use the general approach of mapping function
evaluations to represent both input and output functions $u$ and $v$.

!!! note annotate
    As neural networks take vectors as input, we need to vectorize the
    functions $u$ and $v$ in some sense. We could represent the functions within
    finite-dimensional function spaces (e.g., the space of $n$-th order
    polynomials) and map the coefficients. However, a more general approach is
    to map evaluations of the functions at a finite set of evaluation points.
    This was proposed in the original DeepONet paper and is also used in other
    neural operator architectures.

Let $x_i \in X,\ 1 \leq i \leq n,$ be a finite set of *collocation points*
(or *sensor positions*) in the domain $X$ of $u$.
We represent the function $u$ by its evaluations at these collocation
points and write $\mathbf{x} = (x_i)_i$ and $\mathbf{u} = (u(x_i))_i$.
This finite dimensional representation is fed into the neural operator.

The mapped function $v = G(u)$, on the other hand, is also represented by
function evaluations only. Let $y_j \in Y,\ 1 \leq j \leq m,$ be a set of
*evaluation points* (or *query points*) in the domain $Y$ of $v$ and
$\mathbf{y} = (y_j)_j$.
Then, the output values $\mathbf{v} = (v(y_j))_j$ are approximated by the neural
operator
$$
v(\mathbf{y}) = G(u)(\mathbf{y})
\approx G_\theta(\mathbf{x}, \mathbf{u}, \mathbf{y}) = \mathbf{v}.
$$

In Python, we write the operator call as
```
v = operator(x, u, y)
```
with tensors `x`, `u`, `y`, `v` of shape `[b, n, d]`, `[b, n, c]`, `[b, m, p]`,
and `[b, m, q]`, respectively, and a batch size `b`.
This is to provide the most general case for implementing operators, as
some neural operators differ in the way they handle input and output values.

For convenience, the call can be wrapped to mimic the mathematical syntax.
For instance, for a fixed set of collocation points `x`, we could define
```
G = lambda y: lambda u: operator(x, u, y)
v = G(u)(y)
```

Operators extend the concept of neural networks to function mappings, which
enables discretization-invariant and mesh-free mappings of data with
applications to physics-informed training, super-resolution, and more.

See our examples in [[operators]] for more details and further reading.
