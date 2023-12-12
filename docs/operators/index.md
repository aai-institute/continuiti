---
title: Learning Operators
alias:
  name: operators
  text: Learning Operators
---

# Introduction

## Operators

In mathematics, _operators_ are function mappings – they map functions to functions.

Let $u: \mathbb{R}^d \to \mathbb{R}^c$ be a function that maps a
$d$-dimensional input to $c$ *channels*. Then, an **operator**
$$
G: u \to v
$$
maps $u$ to a function $v: \mathbb{R}^{d'} \to \mathbb{R}^{c'}$.

!!! example annotate
    The operator $G: u \to \partial_x u$ maps functions $u$ to their
    partial derivative $\partial_x u$.

## Learning Operators

Learning operators is the task of learning the mapping $G$ from data.
In the context of neural networks, we want to learn a neural network $G_\theta$
with parameters $\theta$ that, given a set of input-output pairs $(u_k, v_k)$,
maps $u_k$ to $v_k$.

As neural networks take vectors as input, we need to vectorize the input
function $u$ somehow. There are two possibilities:

1. We represent the function $u$ within a finite-dimensional function space
  (e.g. the space of polynomials) and map the coefficients, or
2. We map evaluations of the function at a finite set of evaluation points.

In **Continuity**, we use the second, more geneal approach of mapping function
evaluations, and use this also for the representation of the output function $v$.

In the input domain, we evaluate the function $u$ at a set of points $x_i$ and
collect a set of *sensors* $(x_i, u(x_i))$ in an *observation*
$$
\mathcal{O} = \\{ (x_i, u(x_i)) \mid i = 1, \dots N \\}.
$$

The mapped function can then be evaluated at query points $\mathbf{y}$ to obtain the output
$$
v(\mathbf{y}) = G(u)(\mathbf{y}) \approx G_\theta(\mathbf{x}, \mathbf{u}; \mathbf{y}) = \mathbf{v}
$$
where $\mathbf{x} = (x_i)_i$ and $\mathbf{y} = (y_j)_j$ are the evaluation points
of the input and output domain, respectively, and $\mathbf{u} = (u_i)_i$ is the
vector of function evaluations at $\mathbf{x}$.
The output $\mathbf{v} = (v_j)_j$ is the vector of function evaluations at $\mathbf{y}$.


In Python, this call can be written like
```
v = operator(x, u, y)
```

## Applications to PDEs

Operators are ubiquitous in mathematics and physics. They are used to describe
the dynamics of physical systems, such as the Navier-Stokes equations in fluid
dynamics. As solutions of PDEs are functions, it is natural to use the concept
of neural operators to learn solution operators of PDEs. One possibility to do
this is using an inductive bias, or _physics-informed_ training.
See our examples in [[operators]] for more details.
