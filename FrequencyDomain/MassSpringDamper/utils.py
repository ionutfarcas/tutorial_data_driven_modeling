# utils.py
"""System construction and visualization for the mass-spring-damper example."""

__all__ = [
    "SISO",
    "MassSpringDamper",
    "configure_matplotlib",
    "plot_response",
    "plot_samples",
    "plot_comparison",
    "plot_singular_values"
]

import numpy as np
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


class MassSpringDamper(SISO):
    """Mass-spring-damper SISO system.

    Parameters
    ----------
    num_masses : int
        Number of masses and springs in the system.
        The number of degrees of freedom in the system is twice this number.
    mass : float or (k,) ndarray
        Masses coefficient(s).
        If a float, assume the coefficients are equal for all masses.
    damping : float or (k,) ndarray
        Damping coefficient(s).
        If a float, assume the coefficients are equal for all springs.
    stiffness : float or (k,) ndarray
        Stiffness coefficient(s).
        If a float, assume the coefficients are equal for all springs.
    """

    def __init__(
        self,
        num_masses: int = 3,
        mass = 1.0,
        damping = 0.1,
        stiffness = 10.0,
    ):
        # System order.
        n = 2 * num_masses
        k = num_masses

        # Expand coefficients to arrays if floats were given.
        if not isinstance(mass, np.ndarray):
            mass = mass * np.ones((k,))
        if not isinstance(damping, np.ndarray):
            damping = damping * np.ones((k,))
        if not isinstance(stiffness, np.ndarray):
            stiffness = stiffness * np.ones((k,))

        # Construct second-order matrices.
        M = sparse.diags(mass, offsets = 0, shape = (k, k))
        D = sparse.diags(damping, offsets = 0, shape = (k, k))

        K = sparse.diags(
            [
                np.append(
                    stiffness[0],
                    stiffness[0 : k - 1] + stiffness[1:k],
                ),
                -stiffness[0 : k - 1],
                -stiffness[0 : k - 1],
            ],
            offsets=[0, 1, -1],
            shape=(k, k),
        )

        # Construct first-order matrices.
        A = sparse.block_array([[None, sparse.eye(k)], [-K, -D]])

        b    = np.zeros((n,))
        b[k] = 1.0

        c    = np.zeros((n,))
        c[k] = 1.0

        E = sparse.block_diag([sparse.eye(k), M])

        # Call the parent class constructor to initialize the system.
        SISO.__init__(self, A.tocsc(), b, c, E.tocsc())


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


def plot_samples(leftfreq, leftval, rightfreq, rightval, ax=None):
    """Plot left and right data samples."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 4))

    plot_response(
        leftfreq,
        leftval,
        ax = ax,
        ls = "none",
        marker = "o",
        color = "r",
        label = "Left data",
    )
    plot_response(
        rightfreq,
        rightval,
        ax = ax,
        ls = "none",
        marker = "d",
        color = "k",
        label = "Right data",
    )

    return ax


def plot_comparison(
    frequencies,
    true_responses,
    model_responses,
    relative_errors,
    leftfreq=None,
    leftval=None,
    rightfreq=None,
    rightval=None,
):
    """Plot true frequencies and responses, training data, model responses,
    and model relative error.
    """
    _, axes = plt.subplots(2, 1, sharex=True, figsize = (12, 6))

    # True frequency responses.
    plot_response(
        frequencies, true_responses, ax = axes[0], label = "True response"
    )

    # Training data.
    if leftfreq is not None:
        plot_samples(leftfreq, leftval, rightfreq, rightval, ax = axes[0])

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
