# utils.py
"""Data generation and visualization for the 1D compressible Euler equations.

Run this file to generate the training and testing data used in
demo.ipynb and demo-with-opinf-package.ipynb.
"""

__all__ = [
    "load_experiment_data",
    "configure_matplotlib",
    "plot_initial_conditions",
    "plot_traces",
    "plot_singular_values",
    "plot_basis_vectors",
    "plot_reduced_trajectories",
    "plot_regimes",
]

import os
import h5py
import time
import itertools
import numpy as np
import scipy.integrate
import matplotlib.colors
import matplotlib.animation
import matplotlib.pyplot as plt


DATAFILE = os.path.abspath("full_order_data.h5")


# Full-order model ============================================================
class EulerFiniteDifferenceModel:
    """Finite difference model for the 1D compressible Euler equations
    with periodic boundary conditions.

    Parameters
    ----------
    L : float
        Length of the spatial domain [0, L].
    n_x : int
        Number of points in the spatial discretization.
    """

    n_var = 3  # Number of physical variables.
    gamma = 1.4  # Heat capacity ratio.

    def __init__(self, L: float = 2.0, n_x: int = 200):
        """Define the spatial domain."""
        self.x = np.linspace(0, L, n_x + 1)[:-1]  # Strip x = L due to BCs.
        self.dx = self.x[1] - self.x[0]
        self.L = L

    def spline_initial_conditions(self, init_params):
        r"""Generate initial conditions by evaluating periodic cubic splines
        for density and velocity.

        Parameters
        ----------
        init_params : (6,) ndarray
            Degrees of freedom for the initial conditions, three interpolation
            values for the density and three for the velocity (in that order).

        Returns
        -------
        init : (n_var * n_x,) ndarray
            Initial conditions (in conservative variables [rho, rho v, rho e]).
        """
        x, L = self.x, self.L
        # Extract initial condition parameters and enforce periodicity.
        rho0s, v0s = init_params[0:3], init_params[3:6]
        v0s = np.concatenate((v0s, [v0s[0]]))
        rho0s = np.concatenate((rho0s, [rho0s[0]]))

        # Construct initial conditions for each variable.
        nodes = np.array([0, L / 3, 2 * L / 3, L]) + x[0]
        rho = scipy.interpolate.CubicSpline(nodes, rho0s, bc_type="periodic")(
            x
        )
        v = scipy.interpolate.CubicSpline(nodes, v0s, bc_type="periodic")(x)
        p = 1e5 * np.ones_like(x)
        rho_e = p / (self.gamma - 1) + 0.5 * rho * v**2

        return np.concatenate((rho, rho * v, rho_e))

    def _ddx(self, variable):
        """Estimate d/dx with first-order backward differences,
        assuming periodic boundary conditions.
        """
        return (variable - np.roll(variable, 1, axis=0)) / self.dx

    def derivative(self, tt, state):
        """Right-hand side of the model equations, d/dt of the state."""
        rho, rho_v, rho_e = np.split(state, self.n_var)
        v = rho_v / rho
        p = (self.gamma - 1) * (rho_e - 0.5 * rho * v**2)

        return -np.concatenate(
            [
                self._ddx(rho_v),
                self._ddx(rho * v**2 + p),
                self._ddx((rho_e + p) * v),
            ]
        )

    def solve(self, q_init, time_domain):
        """Integrate the model in time with SciPy."""
        return scipy.integrate.solve_ivp(
            fun=self.derivative,
            t_span=[time_domain[0], time_domain[-1]],
            y0=q_init,
            method="RK45",
            t_eval=time_domain,
            rtol=1e-6,
            atol=1e-9,
        ).y


# Experiment data =============================================================
def generate_experiment_data():
    """Generate training and testing data used in Experiment 1
    (see euler.ipynb and euler-opinf.ipynb).
    """
    print("Generating experiment data")

    # Initialize the full-order model.
    fom = EulerFiniteDifferenceModel(L=2, n_x=200)

    # Construct the full time domain.
    t_final = 0.15
    n_t = 501
    t_all = np.linspace(0, t_final, n_t)

    # Choose a shorter time domain for training data.
    t_obs = 0.06
    t_train = t_all[t_all <= t_obs]

    # Generate several initial conditions for training data.
    q0s = [
        fom.spline_initial_conditions(icparams)
        for icparams in itertools.product(
            [20, 24],
            [22],
            [20, 24],
            [95, 105],
            [100],
            [95, 105],
        )
    ]

    # Solve the full-order model for each training initial condition.
    print(f"{len(q0s)} FOM solves -> training data...", end="", flush=True)
    _start = time.time()
    Q_fom = np.array([fom.solve(q0, t_train) for q0 in q0s])
    _elapsed = time.time() - _start
    print(f"done in {_elapsed:.6f} s")

    # Generate an initial condition for testing.
    test_init = fom.spline_initial_conditions([22, 21, 25, 100, 98, 102])

    # Solve the full-order model for the testing initial condition.
    print("1 FOM solve -> testing data...", end="", flush=True)
    _start = time.time()
    test_solution = fom.solve(test_init, t_all)
    _elapsed = time.time() - _start
    print(f"done in {_elapsed:.6f} s")

    # Save the data.
    with h5py.File(DATAFILE, "w") as hf:
        hf.create_dataset("gamma", data=[EulerFiniteDifferenceModel.gamma])
        for key, dataset in (
            ("x", fom.x),
            ("t_train", t_train),
            ("training_snapshots", Q_fom),
            ("t_all", t_all),
            ("test_init", test_init),
            ("test_solution", test_solution),
        ):
            hf.create_dataset(
                key,
                data=dataset,
                compression="gzip",
                compression_opts=9,
                shuffle=True,
                fletcher32=True,
            )

    print(f"Saved experiment data to {DATAFILE}\n")


def load_experiment_data():
    """Generate training and testing data used in Experiment 1
    (see euler.ipynb and euler-opinf.ipynb).
    """
    if not os.path.isfile(DATAFILE):
        generate_experiment_data()

    print(f"Loading experiment data from {DATAFILE}")
    data = {}
    with h5py.File(DATAFILE, "r") as hf:
        data["gamma"] = hf["gamma"][0]
        for key in (
            "x",
            "t_train",
            "training_snapshots",
            "t_all",
            "test_init",
            "test_solution",
        ):
            data[key] = hf[key][:]
    return data


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


def split_variables(states):
    """Split the overall Euler state into individual physical variables."""
    return np.split(states, EulerFiniteDifferenceModel.n_var, axis=0)


def _euler_ylabels(fig, axes, lifted: bool = False):
    """Add y-labels to Euler state plots."""
    axes[0].set_ylabel(r"$v$" if lifted else r"$\rho$")
    axes[1].set_ylabel(r"$p$" if lifted else r"$\rho v$")
    axes[2].set_ylabel(r"$1/\rho$" if lifted else r"$\rho e$")
    fig.align_ylabels(axes)


def plot_initial_conditions(x, trajectories, ncol: int = 3):
    """Plot three sets of initial conditions over the spatial domain."""
    q0s = [Q[:, 0] for Q in trajectories]

    fig, axes = plt.subplots(
        3,
        ncol,
        figsize=(4 * ncol, 6),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for i, idx in enumerate(
        np.sort(np.random.choice(len(q0s), size=ncol, replace=False))
    ):
        q0 = q0s[idx]
        for ax, var in zip(axes[:, i], split_variables(q0)):
            ax.plot(x, var)
            ax.set_xlim(x[0], x[-1])
        axes[0, i].set_title(f"Initial condition {idx}")
        axes[-1, i].set_xlabel(r"$x\in[0,L]$")
    _euler_ylabels(fig, axes[:, 0])
    fig.tight_layout()

    return fig, axes


def plot_traces(x, t, states, nlocs: int = 20, lifted: bool = False):
    """Plot traces in time at ``nlocs`` locations."""
    # Choose spatial locations at which to plot each state.
    xlocs = np.linspace(0, states.shape[0] // 3, nlocs + 1, dtype=int)[:-1]
    xlocs += xlocs[1] // 2
    colors = plt.cm.twilight(np.linspace(0, 1, nlocs + 1)[:-1])

    # Plot the states in time.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
    for i, c in zip(xlocs, colors):
        for ax, var in zip(axes, split_variables(states)):
            ax.plot(t, var[i], color=c, lw=1)

    # Format axes.
    axes[2].set_xlabel(r"$t\in[t_0,t_{\textrm{final}}]$")
    axes[2].set_xlim(t[0], t[-1])
    _euler_ylabels(fig, axes, lifted)

    # Colorbar.
    lsc = plt.cm.twilight(np.linspace(0, 1, 400))
    scale = matplotlib.colors.Normalize(vmin=0, vmax=1)
    lscmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "euler", lsc, N=nlocs
    )
    mappable = plt.cm.ScalarMappable(norm=scale, cmap=lscmap)
    cbar = fig.colorbar(mappable, ax=axes, pad=0.015)
    cbar.set_ticks(x[xlocs] / (x[-1] - x[0]))
    cbar.set_ticklabels([f"{xx:.2f}" for xx in x[xlocs]])
    cbar.set_label(r"spatial coordinate $x$")

    return fig, axes


def plot_singular_values(sigma, upto: int = 20):
    """Plot all singular values and an up-close"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot all POD singular values.
    idx = np.arange(1, sigma.size + 1)
    axes[0].semilogy(idx, sigma)
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("POD singular values")
    axes[0].set_title("all singular values")

    # Plot just the first few singular values.
    axes[1].semilogy(idx[:upto], sigma[:upto], marker="o", lw=1)
    axes[1].axvline(9, lw=1, color="gray", zorder=0)
    axes[1].set_xlabel("singular value index")
    axes[1].set_title("closeup of dominant singular values")
    axes[1].set_xticks(range(3, upto + 1, 3))

    fig.tight_layout()
    return fig, axes


def plot_basis_vectors(x, basis_matrix, sharey: bool = True):
    """Plot basis vectors over the spatial domain for each state variable."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=sharey)
    for basis_chunk, ax in zip(split_variables(basis_matrix), axes):
        for i in range(basis_matrix.shape[1]):
            ax.plot(x, basis_chunk[:, i], label=rf"basis vector ${i+1}$")
        ax.set_xlim(x[0], x[-1])

    axes[0].set_ylabel("velocity basis vectors")
    axes[1].set_ylabel("pressure basis vectors")
    axes[2].set_ylabel("spec.vol basis vectors")
    axes[2].set_xlabel(r"$x$")

    fig.subplots_adjust(hspace=0.2, right=0.8)
    axes[0].legend(
        loc="center right",
        bbox_to_anchor=(1, 0.5),
        bbox_transform=fig.transFigure,
    )
    fig.align_ylabels(axes)
    plt.show()


def plot_reduced_trajectories(t, trajectories):
    """Plot reduced state coordinates in time for two trajectories."""
    Qhat, Qhat_rom = trajectories

    fig, axes = plt.subplots(3, 3, sharex=True, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(t, Qhat[i, :], "k-", label="training data")
        ax.plot(t, Qhat_rom[i, :], "C0--", label="ROM solution")
        ax.set_ylabel(rf"$\hat{{q}}_{{{i+1}}}(t)$")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$t$")
    fig.tight_layout()

    fig.subplots_adjust(bottom=0.15)
    axes[0, 0].legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
    )

    return fig, axes


def plot_regimes(
    x,
    t_train,
    t_all,
    states,
    nlocs: int = 20,
    lifted: bool = False,
    title: str = "",
):
    """Plot traces in time and label training/prediction regimes."""
    fig, axes = plot_traces(x, t_all, states, nlocs, lifted)

    for ax in axes.flat:
        ax.axvline(t_train[-1], color="black", linewidth=1)
    ymax = axes[0].get_ylim()[1]
    axes[0].text(t_train[-1] / 2, ymax, "training regime", ha="center")
    xpred = (t_all[-1] - t_train[-1]) / 2 + t_train[-1]
    axes[0].text(xpred, ymax, "prediction regime", ha="center")
    if title:
        axes[0].set_title(title, fontsize="large")

    return fig, axes


# =============================================================================
if __name__ == "__main__":
    if os.path.isfile(DATAFILE):
        raise FileExistsError(DATAFILE)
    generate_experiment_data()
