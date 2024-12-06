# utils.py
"""Data loading and visualization for the ..."""

__all__ = [
    "BarycentricModel",
    "load_data",
    "configure_matplotlib",
    "plot_response",
    "plot_comparison",
    "plot_error_history"
]

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# Linear systems and full-order model =========================================
class BarycentricModel:
    """Scalar barycentric form models with
    
               h1*w1 / (s - l1) + ... + hr*wr / (s - lr)
        G(s) = ----------------------------------------
               1 + w1 / (s - l1) + ... + wr / (s - lr)

        Parameters
        ----------
        wi : weights
        hi : frequency response values
        li : barycentric expansion points
    """

    def __init__(self, weights = None, tfvalues = None, tfpoints = None):
        """Store barycentric parameters."""
        self.weights  = weights
        self.tfvalues = tfvalues
        self.tfpoints = tfpoints

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

        if self.weights is None:
            return np.array(
                [0 for s in points]
            )

        A, b, c = self.toStateSpace()
        E       = np.eye(A.shape[0])

        return np.array(
            [c @ np.linalg.solve(s * E - A, b) for s in points]
        )

    def toStateSpace(self):
        """Create state-space model for barycentric form."""
        if self.weights is None:
            return None

        c = self.tfvalues
        b = self.weights
        A = np.diag(self.tfpoints) - b.reshape((len(b), 1)) @ np.ones((1, len(b)))
        
        return A, b, c
    

# Data Import =================================================================
def load_data():
    """Load MAT file into Python variables.

    Returns
    -------
    w   : frequency vector
    mu  : complex evaluation points
    mag : transfer function magnitudes
    tf  : transfer function values
    """
    mat = sp.io.loadmat('porous_bone_data.mat')

    return (
        mat['frequencies'].reshape((len(mat['frequencies'],))),
        mat['responses'].reshape((len(mat['responses'],))),
    )


# Visualization ===============================================================
def configure_matplotlib(latex_is_installed: bool = True):
    """Some minor matplotlib configuration."""
    plt.rc("axes", titlesize="x-large", labelsize="x-large")
    plt.rc("axes.spines", right=False, top=False)
    plt.rc("figure", figsize=(12, 6), dpi=300)
    plt.rc("legend", edgecolor="none", frameon=False, fontsize="xx-large")

    if latex_is_installed:
        plt.rc("font", family="serif")
        plt.rc("text", usetex=True)

def plot_response(
    frequencies: np.ndarray,
    responses: np.ndarray,
    ax=None,
    **kwargs,
):
    """Plot the magnitude of the frequency response."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 4))

    ax.loglog(frequencies, np.abs(responses), **kwargs)
    ax.set_xlabel(r"frequency $\omega$ (rad/s)")
    ax.set_ylabel(r"magnitude $|G(\mathrm{j}\omega)|$")

    return ax

def plot_comparison(
    frequencies,
    responses,
    model_responses,
    relative_errors,
    interpfreq = None,
    interpval = None
):
    """Plot true frequencies and responses, training data, model responses,
    and model relative error.
    """
    _, axes = plt.subplots(2, 1, sharex = True, figsize = (12, 6))

    # True frequency responses.
    plot_response(
        frequencies,
        responses,
        ax        = axes[0],
        linestyle = "",
        marker    = ".",
        label     = "Transfer function data"
    )

    # Model frequency responses.
    plot_response(
        frequencies,
        model_responses,
        ax        = axes[0],
        linestyle = "--",
        label     = "Barycentric model"
    )

    # Expansion points.
    if interpfreq is not None:
        plot_response(
            interpfreq.imag,
            np.abs(interpval),
            ax        = axes[0],
            linestyle = "",
            marker    = "x",
            color     = "k",
            label     = "Expansion points"
        )

    axes[0].legend(loc = "upper left")
    axes[0].set_xlabel("")

    # Relative error.
    plot_response(frequencies, relative_errors, ax = axes[1], color = "k")
    axes[1].set_ylabel("weighted error")

    return axes

def plot_error_history(error):
    """Plot error history of the AAA algorithm."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.semilogy(
        list(range(1, len(error) + 1)),
        error,
        "o-" if len(error) < 20 else "-",
        label = "weighted iteration error"
    )
    ax.legend(loc = "lower left")

    return ax
