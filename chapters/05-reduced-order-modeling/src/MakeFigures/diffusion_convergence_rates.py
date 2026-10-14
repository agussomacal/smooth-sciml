from typing import List, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import figures_path
from src.lib.SolutionsManagers import SolutionsManagerFEM
from src.lib.VizUtils import save_fig
from src.notebooks.functions4notebooks import *


def solution_sampling_method_uniform(N, diffusion_contrast_lower, diffusion_contrast_upper, num_blocks) -> np.ndarray:
    """
    return:
    - array of shape (N, number_of_blocks)
    """
    return np.random.uniform(diffusion_contrast_lower, diffusion_contrast_upper, size=(N, num_blocks))


# ---------- pca algorithm ---------- #
def reduced_basis_generator_pca(solutions_offline: List[np.ndarray], number_of_reduced_base_elements: int) -> List[
    np.ndarray]:
    transformer = StandardScaler(with_mean=True, with_std=True)  # to center by mean and scale by variance.
    # the mean and std is taken along the columns which means on each of the dimensions of the V space.
    basis = PCA(n_components=number_of_reduced_base_elements).fit(
        transformer.fit_transform(solutions_offline)).components_
    # we need to come back to the original space
    return transformer.inverse_transform(basis)


def get_function_norm(norm="l2"):
    """
    Get a function that calculates the desired norm (l2 or h10) when applied on a list of elements of the V space.
    """
    if norm == "l2":
        f_norm = sm.l2norm
    elif norm == "h10":
        f_norm = sm.H10norm
    else:
        raise Exception(f"Norm {norm} not implemented.")
    return f_norm


# ---------- greedy algorithm ---------- #
def reduced_basis_generator_greedy(solutions_offline: List[np.ndarray], number_of_reduced_base_elements: int,
                                   function_norm: Callable) -> List[np.ndarray]:
    basis = [solutions_offline[np.argmax(function_norm(solutions_offline), axis=0)]]
    for _ in range(1, number_of_reduced_base_elements):
        x = np.linalg.lstsq(np.transpose(basis), np.transpose(solutions_offline), rcond=None)[0]
        next_element = np.argmax(function_norm((np.transpose(solutions_offline) - np.transpose(basis) @ x).T))
        basis.append(solutions_offline[next_element])
    return basis


if __name__ == "__main__":
    plt.style.use("figures.mplstyle")
    np.random.seed(42)
    axes_xy_proportions = (4, 4)

    max_vn_dim = 20
    number_of_solutions = 200
    diffusion_contrast_lower = 1
    diffusion_contrast_upper = 100
    sm = SolutionsManagerFEM(blocks_geometry=(2, 2), discretization=15, num_cores=1)
    # diffusion_coefficients = solution_sampling_method_uniform(number_of_solutions, diffusion_contrast_lower,
    #                                                           diffusion_contrast_upper, np.prod(sm.blocks_geometry))
    # diffusion_coefficients = diffusion_coefficients.reshape((-1,) + sm.blocks_geometry)  # shape: N x 2 x 2 = 200 x 2x 2

    diffusion_coefficients = np.random.uniform(diffusion_contrast_lower, diffusion_contrast_upper,
                                               (number_of_solutions,) + sm.blocks_geometry)
    # diffusion_coefficients = np.ones((number_of_solutions, ) + sm.blocks_geometry)
    # diffusion_coefficients[:, 0, 0] = np.random.uniform(diffusion_contrast_lower, diffusion_contrast_upper, number_of_solutions)
    # diffusion_coefficients[:, 0, 1] = np.random.uniform(diffusion_contrast_lower, diffusion_contrast_upper, number_of_solutions)

    solutions = sm.generate_solutions(diffusion_coefficients)  # shape: N x dim(V) = 200 x 361
    reduced_basis_dict = {
        "SVD": reduced_basis_generator_pca(solutions, max_vn_dim),
        "Greedy": reduced_basis_generator_greedy(solutions, max_vn_dim,
                                                 function_norm=get_function_norm("h10")),
    }

    dim_n = np.arange(1, max_vn_dim, dtype=int)
    errors = []
    for rb_method, rb_basis in reduced_basis_dict.items():
        error = []
        for n in tqdm(dim_n):
            basis = rb_basis[:n]
            approximate_solutions = sm.generate_fm_solutions(theta=diffusion_coefficients, basis=basis)
            error.append(error_metrics_dict["Average case: H¹(Ω)"](sm, solutions - approximate_solutions))
        errors.append(error)

    dim_d = np.prod(sm.blocks_geometry)
    # dim_d = 2
    C = 5e3
    c = 40.
    theoretical_curve = C * np.exp(-2 ** (-1 - 2 / dim_d) * c * dim_n ** (1. / dim_d))
    with save_fig(f"{figures_path}/forward_problem_decay_rate.pdf", dpi=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        # ax.plot(dim_n, np.exp(-1.5*dim_n**(1/(2*np.prod(sm.blocks_geometry)-2))), ":k", label="Theoretical trend")
        ax.plot(dim_n, theoretical_curve, ":k", label=r"$d_n^{\text{wc}}(\mathcal{M})$")

        [ax.plot(dim_n, error, ".-", label=rb_method) for rb_method, error in zip(reduced_basis_dict, errors)]
        ax.set_xticks(np.arange(1, max_vn_dim, 2, dtype=int))
        ax.set_xlabel(r"Reduced basis dimension: $\dim(V_n)=n$")
        ax.set_ylabel(r"Average $H^1_0(\Omega)$ error")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()
        fig.tight_layout()
