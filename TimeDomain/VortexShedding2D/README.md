# Vortex-shedding Flow Past a Cylinder (2D)

This is the second demonstration for the time-domain perspective: a mid-sized problem defined on a two-dimensional spatial domain.

## Contents

- [demo.ipynb](./demo.ipynb): Walkthrough using only the standard python scientific stack.
- [demo-with-opinf-package.ipynb.ipynb](./demo-with-opinf-package.ipynb): Walkthrough using the [`opinf`](https://willcox-research-group.github.io/rom-operator-inference-Python3/source/index.html) package for model reduction tasks.
- [utils.py](./utils.py): Auxiliary file defining data generation and visualization routines.

Training data are generated and stored in `full_order_data.h5`.

## Problem Statement

We consider the canonical problem of two-dimensional transient flow past a circular cylinder, governed by the 2D incompressible Navier-Stokes equations

$$
\begin{aligned}
    \partial_t \mathbf{u} + \nabla \cdot (\mathbf{u} \otimes \mathbf{u})
    &= \nabla p + Re^{-1}\Delta \mathbf{u}
    \\
    \nabla \cdot \mathbf{u}
    &= 0,
\end{aligned}
$$

where $p \in \mathbb{R}$ denotes the pressure, $\mathbf{u} = (u_x, u_y)^\mathsf{T} \in \mathbb{R}^2$ denotes the $x$ and $y$ components of the velocity vector, and $Re$ denotes the dimensionless Reynolds number.

The problem setup, geometry, and parameterization are based on the [DFG 2D-3 benchmark in the FeatFlow suite](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html).
This setup uses $Re = 100$, which represents a value that is above the critical Reynolds number for the onset of the two-dimensional vortex shedding the physical domain is depicted in the figure below:
<img src="./geometry.png" width="700" class="center">
