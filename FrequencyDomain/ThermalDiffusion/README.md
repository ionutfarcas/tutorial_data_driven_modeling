# Thermal Diffusion

This is the second demonstration for the frequency-domain perspective: a problem where the PDE transfer function is given (but not the discretization), approached with the Loewner framework.

## Contents

- [demo.ipynb](./demo.ipynb): Demonstration notebook.
- [utils.py](./utils.py): Auxiliary file defining data generation and visualization routines.

## Problem Statement

In this example, we consider the one-dimensional heat equation with radiation of the form

$$
\tag{5.1}
\frac{\partial T}{\partial t} = \kappa \frac{\partial^{2} T}{\partial x^{2}} - \beta T,
$$

with $\kappa = \frac{\lambda}{\rho \gamma}$, where $\lambda$ is the thermal conductivity, $\rho$ is the density and $\gamma$ is the specific heat capacity of the material, and $\beta T$ models the heat loss due to radiation. The solution $T(t, x)$ models the heat in the one-dimensional domain. For the boundary of the domain $x \in [x_{\min}, x_{\max}]$, we have external forcing on the left side and temperature observations on the right

$$
T(t, x_{\min}) = u(t) \quad \text{and} \quad y(t) = T(t, x_{\max}).
$$

This linear partial differential equation $(5.1)$ with the boundary condition and observation can be equivalently be described in the frequency domain via

$$ \tag{5.2}
Y(s) = G(s) U(s) = \underbrace{\left( e^{-a \sqrt{(s + \mu)\kappa^{-1}}} \right)}_{G(s)} U(s),
$$

where $a = x_{\max} - x_{\min}$ is the length of the domain, $Y(s)$ and $U(s)$ are the Laplace transforms of the heat inflow and the observagtions, and $G(s)$ is the systems transfer function.

The goal in this example is to use evaluations the transfer function in $(5.2)$ to derive a reduced-order linear time-invariant finite-dimensional system of ordinary differential equations that describe the dynamics of the partial differential equation $(5.1)$ without constructing an intermediate discretization.
The advantage of working with the transfer function is that the discretization that we create is directly targeted towards the input-to-output (heat inflow to observation) behavior of the dynamical system $(5.1)$ without the intermediate step of describing the complete internal system behavior. The finite-dimensional reduced-order model will be given via the transfer function

$$ \tag{5.3}
\widehat{G}(s) = \widehat{\mathbf{c}}^{\mathsf{T}} (s \widehat{\mathbf{E}} - \widehat{\mathbf{A}})^{-1} \widehat{\mathbf{b}},
$$

which we can interprete in the time domain as a system of generalized ordinary differential equations of the form

$$
\begin{aligned}
  \widehat{\mathbf{E}} \dot{\widehat{\mathbf{q}}}(t) & = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t) + \widehat{\mathbf{b}} u(t), \\
  \widehat{y}(t) & = \widehat{\mathbf{c}}^{\mathsf{T}} \widehat{\mathbf{q}}(t),
\end{aligned}
$$

with $\widehat{\mathbf{A}}, \widehat{\mathbf{E}} \in \mathbb{R}^{r \times r}$, $\widehat{\mathbf{b}}, \widehat{\mathbf{c}} \in \mathbb{R}^{r}$ and the state-space dimension $r$ small.
