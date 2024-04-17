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

In mathematics, _operators_ are function mappings: they map functions to
functions. Let

$$
U = \{ u: X \subset \mathbb{R}^{d_x} \to \mathbb{R}^{d_u} \}
$$

be a set of functions that map a $d_x$-dimensional input to an $d_u$-dimensional
output, and

$$
V = \{ v: Y \subset \mathbb{R}^{d_y} \to \mathbb{R}^{d_v} \}
$$

be a set of functions that map a $d_y$-dimensional input to a $d_v$-dimensional
output.


An **operator**

\begin{align*}
  G: U &\to V, \\
     u &\mapsto v,
\end{align*}

maps functions $u \in U$ to functions $v \in V$.

!!! example annotate
    The operator $G(u) = \partial_x u$ maps functions $u$ to their
    partial derivative $\partial_x u$.

## Learning Operators

**Operator learning** is the task of learning the mapping $G$ from data.
In the context of neural networks, we want to train a neural network $G_\theta$
with parameters $\theta$ that, given a set of input-output pairs
$(u_k, v_k) \in U \times V$, maps $u_k$ to $v_k$.
We generally refer to such a neural network $G_\theta$ as a *neural operator*.

## Discretization

In **continuiti**, we use the general approach of mapping function
evaluations to represent both input and output functions $u$ and $v$ in
a discretized form.

Let $x_i \in X$ be `num_sensors` many
*sensor positions* (or *collocation points*) in the input domain $X$ of $u$.
We represent the function $u$ by its evaluations at these sensors
and write $\mathbf{x} = (x_i)_i$ and $\mathbf{u} = (u(x_i))_i$.
This finite dimensional representation is fed into the neural operator.

The mapped function $v = G(u)$, on the other hand, is also represented by
function evaluations only. Let $y_j \in Y$ be `num_evaluations`
many *evaluation points* (or *query points*) in the input domain $Y$ of
$v$ and $\mathbf{y} = (y_j)_j$.
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
with tensors of shapes

```
x: (batch_size, x_dim, num_sensors...)
u: (batch_size, u_dim, num_sensors...)
y: (batch_size, y_dim, num_evaluations...)
v: (batch_size, v_dim, num_evaluations...)
```

This is to provide the most general case for implementing operators, as
some neural operators differ in the way they handle input and output values.

## Wrapping
For convenience, the call can be wrapped to mimic the mathematical syntax.
For instance, for a fixed set of collocation points `x`, we could define
```
G = lambda y: lambda u: operator(x, u(x), y)
v = G(u)(y)
```

## Applications
Neural operators extend the concept of neural networks to function mappings,
which enables discretization-invariant and mesh-free mappings of data with
applications to physics-informed training, super-resolution, and more.
See our <a href="../examples">Examples</a> section for more on this.

## Further Reading
Follow our introduction to <a href="../examples/functions">Functions</a> in continuiti
and proceed with the <a href="../examples/training">Training</a> example to learn
more about operator learning in continuiti.
