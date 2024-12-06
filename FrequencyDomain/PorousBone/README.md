# Porous Bone

This is the third demonstration for the frequency-domain perspective: an example with given data (but no transfer function or discretization), approached with the AAA algorithm.

## Contents

- [demo.ipynb](./demo.ipynb): Demonstration notebook.
- [utils.py](./utils.py): Auxiliary file defining data generation and visualization routines.
- `porous_bone_data.mat`: File containing the transfer function data for the example.

## Problem Statement

In this example, we have given frequency domain data for the vibrational response of a porous bone. This data has our usual duplet form

$$
(\mathfrak{j} \omega_{1}, g_{1}), \quad (\mathfrak{j} \omega_{2}, g_{2}), \quad \ldots, \quad (\mathfrak{j} \omega_{N}, g_{N}),
$$

where $\omega_{i} \in \mathbb{R}$ are the real frequencies and $g_{i} = G(\mathfrak{j} \omega_{i})$ are evaluations of the unknown transfer functions on the imaginary axis; $\mathfrak{j} = \sqrt{-1}$ is the imaginary unit. Our goal is to find a low-dimensional first-order model of the form

$$ \tag{6.1}
\widehat{G} = \widehat{\mathbf{c}}^{\mathsf{T}} (s \widehat{\mathbf{E}} - \widehat{\mathbf{A}})^{-1} \widehat{\mathbf{b}},
$$

with the matrices $\widehat{\mathbf{E}}, \widehat{\mathbf{A}} \in \mathbb{C}^{r \times r}$, $\widehat{\mathbf{c}}, \widehat{\mathbf{b}} \in \mathbb{C}^{r}$ and reasonably small order $r \ll N$ so that this new transfer function approximates the given data well

$$
  \widehat{G}(\mathfrak{j} \omega_{1}) \approx g_{1}, \quad \widehat{G}(\mathfrak{j} \omega_{2}) \approx g_{2}, \quad \ldots, \quad \widehat{G}(\mathfrak{j} \omega_{N}) \approx g_{N}.
$$

In the previous examples, we used the Loewner framework to create a reduced-order model for the given data with an interpolating transfer function. There are situations in which this may not the desired approach:
* In case of much data ($N$ large), interpolation leads to high-dimensional models.
* While empirically an approximate least-squares behavior has been observed for the Loewner framework when using rank truncations, there are no theoretical guarantees for this. Additionally, the error behavior of the Loewner framework when choosing different reduced orders $r$ is not predictable so that one typically computes differently sized approximations and verifies the errors with respect to the given data.
* When parts of the data are noisy, interpolation of the whole data is typically undesired.

Therefore, we consider a different data-driven modeling method for this example: the Adaptive Antoulas-Anderson (AAA) algorithm. In this method, the reduced-order model is constructed iteratively where in every step only the data corresponding to the worst case approximation errors is interpolated while the rest is fitted by solving a linear least-squares problem. The key to the AAA algorithm is the reformulation of the rational transfer function $(6.1)$ into the interpolating barycentric form

$$ \tag{6.2}
\widehat{G}(s) = \frac{\sum\limits_{i = 1}^{r} \frac{w_{i} h_{i}}{s - \lambda_{i}}}{1 + \sum\limits_{i = 1}^{r} \frac{w_{i}}{s - \lambda_{i}}},
$$

with the barycentric weights $w_{i} \in \mathbb{C}$, the transfer function values $h_{i} \in  \mathbb{C}$ and the expansion points $\lambda_{i} \in \mathbb{C}$.
By construction, the barycentric form satisfies the following interpolation conditions

$$
\widehat{G}(\lambda_{i}) = h_{i} \quad\text{if}~w_{i} \neq 0.
$$

This means that after selecting suitable interpolation points, we have the freedom of choosing the weights $w_{i}$ to improve the approximation quality of the transfer function. In AAA, the weights $w_{i}$ are selected as the solution to a linearized least-sqares problem.
