# utils.py
"""System classes and visualization for the thermal diffusion example."""

__all__ = [
    "SISO",
    "ThermalDiffusion",
    "configure_matplotlib",
    "plot_response",
    "plot_comparison",
    "plot_singular_values",
    "plot_eigenvalues",
    "plot_simulation"
]

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# Linear systems and full-order model =========================================
class SISO:
    """Single-input single-output system of the form

        E q'(t) = A q(t) + b u(t),
           y(t) = c^T q(t),

    where A and E are n x n matrices and b and c are n-dimensional vectors.
    Here, q(t) is the n-dimensional state and u(t) is the scalar input.

    Parameters
    ----------
    A : (n, n) ndarray or scipy.sparse array
        State matrix.
    b : (n,) ndarray
        Input vector.
    c : (n,) ndarray
        Output vector.
    E : (n, n) ndarray or scipy.sparse array or None
        Mass matrix. If None, set E = I, the n x n identity matrix.
    """

    def __init__(self, A, b, c, E=None):
        """Store system matrices."""
        if sparse.issparse(A):
            A = A.tocsc()  # Convert to a sparse format good for linear solves.
        if E is None:
            E = sparse.eye(A.shape[0])
        if sparse.issparse(E):
            E = E.tocsc()
        self.A = A
        self.b = b
        self.c = c
        self.E = E

    def transfer_function(self, points):
        """Evaluate the transfer function G(s) = c^T (s E - A)^-1 B.

        Parameters
        ----------
        points : (N_f,) complex ndarray
            Complex points at which to evaluate the system.
            Usually sqrt(-1) * real_frequencies, but may be more general.

        Returns
        -------
        responses : (N_f,) complex ndarray
            Transfer function values.
        """
        is_sparse = sparse.issparse(self.E) and sparse.issparse(self.A)
        linsolve  = spla.spsolve if is_sparse else la.solve

        return np.array(
            [self.c @ linsolve(s * self.E - self.A, self.b) for s in points]
        )
    
    def time_simulation(self, t_span, q0, input):
        """Compute a time simulation.
        
        Parameters
        ----------
        t_span : (2,) float ndarray
            Starting and final time points.
        q0 : (n,) float ndarray
            Initial value for the simulation.
        input: function of time
            Input signal u(t).
        """

        simulation = sp.integrate.solve_ivp(
            fun    = lambda t, x: 
                np.linalg.solve(self.E, self.A @ x + self.b * input(t)),
            t_span = t_span,
            y0     = q0,
            method = 'RK45',
            rtol   = 1.0e-6,
            atol   = 1.0e-9
        )

        return (simulation["t"], self.c @ simulation["y"])

    
class ThermalDiffusion:
    """Frequency domain thermal diffusion model.

    Parameters
    ----------
    length : float
        Length of the considered domain for the thermal diffusion process.
    conductivity : float
        Thermal conductivity of the material.
    density : float
        Density of the material.
    capacity : float
        Specific heat capacity of the material.
    radiation : float
        Radiation coefficient.
    """

    def __init__(self,
                 length,
                 conductivity,
                 density,
                 capacity,
                 radiation
    ):
        self.length       = length
        self.conductivity = conductivity
        self.density      = density
        self.capacity     = capacity
        self.radiation    = radiation

    def transfer_function(self, points):
        """Evaluate the transfer function G(s).

        Parameters
        ----------
        points : (N_f,) complex ndarray
            Complex points at which to evaluate the system.
            Usually sqrt(-1) * real_frequencies, but may be more general.

        Returns
        -------
        responses : (N_f,) complex ndarray
            Transfer function values.
        """

        return np.array(
            [np.exp(-self.length * np.sqrt((s + self.radiation) 
                * (self.density * self.capacity) / self.conductivity)) 
                for s in points]
        )


# Visualization ===============================================================
def configure_matplotlib(latex_is_installed: bool = True):
    """Some minor matplotlib configuration."""
    plt.rc("axes", titlesize = "x-large", labelsize = "x-large")
    plt.rc("axes.spines", right = False, top = False)
    plt.rc("figure", figsize = (12, 6), dpi = 300)
    plt.rc("legend", edgecolor = "none", frameon = False, fontsize = "xx-large")

    if latex_is_installed:
        plt.rc("font", family = "serif")
        plt.rc("text", usetex = True)

def plot_response(
    frequencies: np.ndarray,
    responses: np.ndarray,
    ax = None,
    **kwargs,
):
    """Plot the magnitude of the frequency response."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize = (12, 4))

    ax.loglog(frequencies, np.abs(responses), **kwargs)
    ax.set_xlabel(r"frequency $\omega$ (rad/s)")
    ax.set_ylabel(r"magnitude $|G(\mathrm{j}\omega)|$")

    return ax

def plot_comparison(
    frequencies,
    true_responses,
    model_responses,
    relative_errors
):
    """Plot true frequencies and responses, training data, model responses,
    and model relative error.
    """
    _, axes = plt.subplots(2, 1, sharex=True, figsize = (12, 6))

    # True frequency responses.
    plot_response(
        frequencies, true_responses, ax = axes[0], label = "True response"
    )

    # Model frequency responses.
    plot_response(
        frequencies, model_responses, ax = axes[0], label = "Loewner", ls = "--"
    )

    axes[0].legend(loc = "upper left")
    axes[0].set_xlabel("")

    # Relative error.
    plot_response(frequencies, relative_errors, ax = axes[1], color = "k")
    axes[1].set_ylabel("relative error")

    return axes

def plot_singular_values(s1, s2):
    """Plot singular value decay for the Loewner pencil matrices."""
    vstacklabel = (
        r"$\left[\begin{array}{l}\mathbf{L}_s \\ \mathbf{L}\end{array}\right]$"
    )
    hstacklabel = r"$[~\mathbf{L}_s~~\mathbf{L}~]$"

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.semilogy(
        list(range(1, len(s1) + 1)),
        s1,
        "o-" if len(s1) < 20 else "-",
        label = f"singular values of {vstacklabel}",
    )
    ax.semilogy(
        list(range(1, len(s2) + 1)),
        s2,
        "s--" if len(s2) < 20 else "--",
        label = f"singular values of {hstacklabel}",
    )
    smax = max(len(s1), len(s2))
    if smax < 20:
        ax.set_xticks(list(range(1, smax + 1)))
        ax.legend(loc="lower left")
    else:
        ax.legend(loc="upper right")

    return ax

def plot_eigenvalues(eigs):
    """Plot eigenvalues in the complex plane."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(
        eigs.real, 
        eigs.imag, 
        "x",
        label = "eigenvalues"
    )
    ymin, ymax = ax.get_ylim()
    ax.plot(
        [0, 0],
        [ymin, ymax],
        "-", 
        label = "imaginary axis"
    )
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.legend(loc = "upper center")

    return ax

def plot_simulation(simulation, **kwargs):
    """Plot time domain simulation"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(
        simulation[0], 
        simulation[1], 
        "-",
        **kwargs
    )
    ax.set_xlabel("time $t$")
    ax.set_ylabel("Output $y(t)$")

    return ax
