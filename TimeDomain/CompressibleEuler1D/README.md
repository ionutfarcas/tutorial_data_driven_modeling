# Compressible Euler Flow of an Ideal Gas (1D)

This is the first demonstration for the time-domain perspective: a small model problem defined on a one-dimensional spatial domain.

## Contents

- [demo.ipynb](./demo.ipynb): Walkthrough using only the standard python scientific stack.
- [demo-with-opinf-package.ipynb.ipynb](./demo-with-opinf-package.ipynb): Walkthrough using the [`opinf`](https://willcox-research-group.github.io/rom-operator-inference-Python3/source/index.html) package for model reduction tasks.
- [utils.py](./utils.py): Auxiliary file defining data generation and visualization routines.

Training data are generated and stored in `full_order_data.h5`.

## Problem Statement

Let $\Omega = [0,L]\subset \mathbb{R}$ be the spatial domain indicated by the variable $x$, and let $[t_0,t_\text{final}]\subset\mathbb{R}$ be the time domain with variable $t$. We consider the one-dimensional Euler equations for the compressible flow of an ideal gas with periodic boundary conditions.
The state is given by

$$
\begin{aligned}
    \vec{q}_\text{c}(x, t) = \left[\begin{array}{c}
        \rho \\ \rho v \\ \rho e
    \end{array}\right],
\end{aligned}
$$

where $\rho = \rho(x,t)$ is the density $[\frac{\text{kg}}{\text{m}^3}]$, $v = v(x,t)$ is the fluid velocity $[\frac{\text{m}}{\text{s}}]$, and $e = e(x, t)$ is the internal energy per unit mass $[\frac{\text{m}^2}{\text{s}^2}]$.
The state evolves according in time according to the following conservative system of partial differential equations (PDEs):

$$
\tag{1.1}
\begin{aligned}
    \frac{\partial\vec{q}_\text{c}}{\partial t}
    = \frac{\partial}{\partial t} \left[\begin{array}{c}
        \rho \\ \rho v \\ \rho e
    \end{array}\right]
    &= -\frac{\partial}{\partial x}\left[\begin{array}{c}
        \rho v \\ \rho v^2 + p \\ (\rho e + p) v
    \end{array}\right]
    & x &\in\Omega,\quad t\in[t_0,t_\text{final}],
    \\
    \vec{q}_\text{c}(0,t) &= \vec{q}_\text{c}(L,t)
    & t &\in[t_0,t_\text{final}],
    \\
    \vec{q}_\text{c}(x,t_0) &= \vec{q}_{\text{c},0}(x)
    & x &\in \Omega,
\end{aligned}
$$

where $p = p(x,t)$ is the pressure $[\text{Pa}] = [\frac{\text{kg}}{\text{m}\cdot\text{s}^2}]$ and $\vec{q}_{\text{c},0}(x)$ is a given initial condition.
The state variables are related via the ideal gas law

$$
\tag{1.2}
\begin{aligned}
    \rho e = \frac{p}{\gamma - 1} + \frac{1}{2}\rho v^{2},
\end{aligned}
$$

where $\gamma = 1.4$ is the nondimensional heat capacity ratio.
