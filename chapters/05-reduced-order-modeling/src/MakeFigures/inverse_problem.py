import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

# Add to PYTHONPATH the folders of the project so the scripts can be imported
this_file_path = __file__
chapter_name = "05-reduced-order-modeling"
chapter_path = os.path.abspath(os.path.join(this_file_path.split(chapter_name)[0], chapter_name))
chapter_src_path = os.path.abspath(os.path.join(chapter_path, "src"))
sys.path.insert(1, chapter_path)
sys.path.insert(1, chapter_src_path)
print(chapter_src_path)

from src.MakeFigures.diffusion_convergence_rates import solution_sampling_method_uniform, reduced_basis_generator_pca
from src.MakeFigures.example_of_diffusion import sm, particular_solution_of_poisson
from src.config import figures_path
from src.lib.VizUtils import SOLUTION_CMAP, prepare_solution4plot, plot_solution, save_fig


def plot_element_of_V(sm, measurement_points, plotname, element_of_V, contour_levels=0, plot_points=True):
    with save_fig(f"{figures_path}/{plotname}.pdf", dpi=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        x, y, u = prepare_solution4plot(sm, element_of_V.T, num_points_per_dim_to_plot)
        u = u.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot))
        plot_solution(ax, x, y, u, sm, contour_levels=contour_levels, vmin=None,
                      vmax=None, colorbar=False, cmap=SOLUTION_CMAP, add_grid=True)
        ax.set_axis_off()

        # plot point
        if plot_points:
            for point_x, point_y in measurement_points:
                ax.scatter(point_x, point_y, c="white", s=10)


def plot_riesz_representers(sm, measurement_points, sigma, solution=None, figname="", vn_basis=None, contour_levels=0):
    def calculate_gauss_riesz(sm, measurement_points, sigma):
        functional = lambda quadrature_points: np.exp(
            -cdist(measurement_points, quadrature_points) ** 2 / 2 / sigma ** 2)
        return sm.calculate_riesz(functional)

    riesz = calculate_gauss_riesz(sm, measurement_points, sigma).T  # (N, m)
    if solution is not None:
        plot_element_of_V(sm, measurement_points, plotname=f"{figname}solution_sigma{sigma}", element_of_V=solution,
                          contour_levels=contour_levels, plot_points=True)

        riesz_proj_coeffs = sm.project_solutions(solution, riesz, coefficients=True).squeeze()
        riesz_proj = sm.project_solutions(solution, riesz, coefficients=False)

        if vn_basis is not None:
            vn_approx = sm.project_solutions(solution, vn_basis, coefficients=False)
            Pw_vn = np.linalg.solve(sm.H10_projection(riesz, riesz), sm.H10_projection(vn_approx, riesz).T).T @ riesz
            pbdw_approx = riesz_proj + vn_approx - Pw_vn

            plot_element_of_V(sm, measurement_points, plotname=f"{figname}vnapprox_sigma{sigma}",
                              element_of_V=vn_approx, contour_levels=contour_levels, plot_points=True)
            plot_element_of_V(sm, measurement_points, plotname=f"{figname}pbdwapprox_sigma{sigma}",
                              element_of_V=pbdw_approx, contour_levels=contour_levels, plot_points=True)

    else:
        riesz_proj_coeffs = np.ones(riesz.shape[0])
        riesz_proj = riesz_proj_coeffs @ riesz

    # plot riez representer of gaussian
    plot_element_of_V(sm, measurement_points, plotname=f"{figname}riesz_sigma{sigma}", element_of_V=riesz_proj,
                      contour_levels=contour_levels, plot_points=True)

    # plot original gaussian
    x, y = np.meshgrid(np.linspace(*sm.x_domain, num=num_points_per_dim_to_plot),
                       np.linspace(*sm.y_domain, num=num_points_per_dim_to_plot))
    points = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    li = np.array([np.exp(-cdist(mp[None, :], points) ** 2 / 2 / sigma ** 2).reshape(
        (num_points_per_dim_to_plot, num_points_per_dim_to_plot)) for mp in measurement_points])
    plot_element_of_V(sm, measurement_points, plotname=f"{figname}riesz_sigma{sigma}", element_of_V=riesz_proj,
                      contour_levels=contour_levels, plot_points=True)
    u = np.einsum("i,i...", riesz_proj_coeffs, li)
    with save_fig(f"{figures_path}/{figname}gauss_sigma{sigma}.pdf", dpi=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        plot_solution(ax, x, y, u, sm, contour_levels=contour_levels,
                      vmin=None, vmax=None, colorbar=False, cmap=SOLUTION_CMAP, add_grid=True)
        ax.set_axis_off()
        # plot point
        for point_x, point_y in measurement_points:
            ax.scatter(point_x, point_y, c="white", s=10)


if __name__ == "__main__":
    plt.style.use(f"{Path(__file__).parent}/figures.mplstyle")
    np.random.seed(42)
    axes_xy_proportions = (4, 4)
    num_points_per_dim_to_plot = 100

    measurement_points = np.array([
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
    ])

    diffusion_coefficients = solution_sampling_method_uniform(
        N=200, diffusion_contrast_lower=1, diffusion_contrast_upper=10, num_blocks=np.prod(sm.blocks_geometry))
    diffusion_coefficients = diffusion_coefficients.reshape((-1,) + sm.blocks_geometry)  # shape: N x 2 x 2 = 200 x 2x 2
    vn_basis = reduced_basis_generator_pca(solutions_offline=sm.generate_solutions(diffusion_coefficients),
                                           number_of_reduced_base_elements=2)

    plot_riesz_representers(sm, measurement_points=np.array([[0.5, 0.5]]), sigma=0.1, solution=None, figname="")
    plot_riesz_representers(sm, measurement_points=np.array([[0.5, 0.5]]), sigma=0.25, solution=None, figname="")
    plot_riesz_representers(sm, measurement_points=measurement_points, sigma=0.1,
                            solution=particular_solution_of_poisson, figname="Projection_")
    plot_riesz_representers(sm, measurement_points=measurement_points, sigma=0.1,
                            solution=particular_solution_of_poisson, vn_basis=vn_basis, contour_levels=10,
                            figname="Approximations_")
