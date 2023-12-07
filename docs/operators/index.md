---
title: Learning Operators
alias:
  name: operators
  text: Learning Operators
---

# Introduction

## Operators

Operators (in mathematics) are function mappings – they map functions to functions.

Let $u: \mathbb{R}^d \to \mathbb{R}^c$ be a function that maps a
$d$-dimensional input to $c$ *channels*. An **operator**
$$
G: u \to v
$$
maps $u$ to a function $v: \mathbb{R}^{d'} \to \mathbb{R}^{c'}$, which is
potentially defined on different domains.

!!! example annotate
    The operator $G: u \to \partial_x u$ maps functions $u$ to their
    partial derivative $\partial_x u$.

## Learning Operators

Learning operators is the task of learning the mapping $G$ from data.
In the context of neural networks, we want to learn a neural network $G_\theta$
with parameters $\theta$ that, given a set of input-output pairs $(u_k, v_k)$,
maps $u_k$ to $v_k$.

As neural networks take vectors as input, we need to vectorize the input
function $u$ somehow. One possibility is to represent the function $u$ within
a finite-dimensional function space, such as the space of polynomials, and
map the coefficients. Another, more general, possibility is to map evaluations
of the function at a set of evaluation points. In **Continuity**, we use the
latter approach.

In the input domain, we evaluate the function $u$ at a set of points $x_i$ and
collect a set of *sensors* $(x_i, u_i)$ in an *observation*
$$
\mathcal{O} = \\{ (x_i, u_i) \mid i = 1, \dots N \\}
$$
where $u_i = u(x_i)$.

The mapped function can then be evaluated at query points
$y$ to obtain the output
$$
v(y) = G_\theta(\mathcal{O})(y).
$$

## Applications to PDEs

Operators are ubiquitous in mathematics and physics. They are used to describe
the dynamics of physical systems, such as the Navier-Stokes equations in fluid
dynamics.
