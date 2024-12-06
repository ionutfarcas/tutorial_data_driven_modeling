# Mass-Spring-Damper System

This is the first demonstration for the frequency-domain perspective: a small test example for the Loewner framework.

## Contents

- [demo.ipynb](./demo.ipynb): Demonstration notebook.
- [utils.py](./utils.py): Auxiliary file defining data generation and visualization routines.

## Problem Statement

We consider a classical mechanical mass-spring-damper system with a single chain of masses connected by springs and independent dampers on every mass. This system with $k$ masses can be described by second-order generalized differential equations of the form

$$
\tag{4.1}
\begin{aligned}
    m_{1} \ddot{p}_{1}(t) + \gamma_{1} \dot{p}_{1}(t) + \kappa_{1}(p_{1}(t) - p_{2}(t))
    &= u(t),
    \\
    m_{i} \ddot{p}_{i}(t) + \gamma_{i} \dot{p}_{i}(t) + \kappa_{i} (p_{i}(t) - p_{i + 1}(t)) - \kappa_{i-1} (p_{i - 1}(t) - p_{i}(t))
    &= 0 \quad \text{for}~~i = 1, \ldots, k - 1,
    \\
    m_{k} \ddot{p}_{k}(t) + \gamma_{k} \dot{p}_{k}(t) + \kappa_{p} p_{k}(t) - \kappa_{k-1} (p_{k - 1}(t) - p_{k}(t))
    &= 0,
\end{aligned}
$$

with mass, damping, and spring stiffness parameters $m_{i}, \gamma_{i}, \kappa_{i} > 0$, for $i = 1, \ldots, k$.
The external forcing $u(t)$ acts only on the first mass.
The quantity of interest is the displacement of the final mass in the chain,

$$
    y(t) = p_{k}(t).
$$

Using the extended state

$$
    \mathbf{q}(t) = \begin{bmatrix}
        p_1(t) \\ \vdots \\ p_k(t)
        \\
        \dot{p}_1(t) \\ \vdots \\ \dot{p}_k(t)
    \end{bmatrix} \in \mathbb{R}^{2k},
$$

the dynamical system $(4.1)$ can be written in the classical linear form

$$
\tag{4.2}
\begin{aligned}
    \mathbf{E} \dot{\mathbf{q}}(t)
    &= \mathbf{A} \mathbf{q}(t) + \mathbf{b} u(t),
    \\
    y(t)
    &= \mathbf{c}^\mathsf{T} \mathbf{q}(t)
\end{aligned}
$$

where $\mathbf{E},\mathbf{A}\in\mathbb{R}^{n\times n}$ and $\mathbf{b},\mathbf{c}\in\mathbb{R}^{n}$ with $n = 2k$.
Because the input $u(t)$ and the output $y(t)$ are both one-dimensional, this is a *single-input single-output* (SISO) system.

The *transfer function* corresponding to $(4.2)$ is given by

$$
\tag{4.3}
\begin{aligned}
    G(s) = \mathbf{c}^\mathsf{T} (s \mathbf{E} - \mathbf{A})^{-1} \mathbf{b},
\end{aligned}
$$

which satisfies $Y(s) = G(s)U(s)$ where $U(s)$ and $Y(s)$ are the Laplace transforms of $u(t)$ and $y(t)$, respectively.

For given measurements of the transfer function $G(s)$, our objective is to find potentially low-dimensional matrices $\hat{\mathbf{A}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}, \hat{\mathbf{E}}$ so that the corresponding transfer function

$$
\widehat{G}(s)
= \widehat{\mathbf{c}}^\mathsf{T} (s \widehat{\mathbf{E}} - \widehat{\mathbf{A}})^{-1} \widehat{\mathbf{b}}
$$

approximates the given data well.
