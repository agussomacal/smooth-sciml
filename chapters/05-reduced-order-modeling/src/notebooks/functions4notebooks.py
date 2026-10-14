import itertools
from collections import defaultdict
from functools import partial
from typing import Tuple

import ipywidgets
import ipywidgets as widgets
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from IPython.core.display_functions import clear_output
from ipywidgets import GridspecLayout
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.lib.SolutionsManagers import SolutionsManager
from src.lib.VizUtils import plot_solutions_together, plot_solution, prepare_solution4plot, SOLUTION_CMAP

# # plot of christoffel function
# x, y = np.meshgrid(sm.points_r[1:-1], sm.points_c[1:-1])
# measurement_points = np.concatenate([x.reshape((-1,1)), y.reshape((-1,1))], axis=1)
# weights = inverse_christoffel_function(reduced_basis_dict["PCA"], sm, measurement_points)
# weights /= np.sum(weights)
#
#
# plt.imshow(np.log10(weights.reshape(19, 19)))

# # plot energy and clutering of reduced basis
# import pandas as pd
# # fig, ax = plt.subplots(figsize=(10, 10))
# basis=reduced_basis_dict["PCA"]
# c = sm.get_Vn_coeffs(theta=diffusion_coefficients, basis=basis)
# np.mean(c**2, axis=0)
# sns.barplot(c**2, log="y")
# sns.clustermap(pd.DataFrame(c).corr().abs(), annot=True, fmt=".1f", linewidth=.5)

error_metrics_dict = {
    "Average case: L²(Ω)": lambda sm, v: np.mean(sm.l2norm(v)),
    "Worst case: L²(Ω)": lambda sm, v: np.max(sm.l2norm(v)),
    "Average case: H¹(Ω)": lambda sm, v: np.mean(sm.H10norm(v)),
    "Worst case: H¹(Ω)": lambda sm, v: np.max(sm.H10norm(v)),
}


def reshape_diffusion_coefficients_to_viz(diffusion_coefficients, blocks_geometry: Tuple[int, int]):
    """

    :param diffusion_coefficients: as given in visualization interface.
    :return: k x block_geometry np.array needed for generate_solutions etc
    """
    return np.array([list(diffusion_coefficients)]).reshape((1,) + blocks_geometry)[:, ::-1]


class Visualizer:
    """
    Defined in case of future generalizations.
    """

    def __init__(self, sm: SolutionsManager):
        self.sm = sm

    def reshape_diffusion_coefficients_to_viz(self, diffusion_coefficients):
        """

        :param diffusion_coefficients: as given in visualization interface.
        :return: k x block_geometry np.array needed for generate_solutions etc
        """
        return reshape_diffusion_coefficients_to_viz(diffusion_coefficients.values(), self.sm.blocks_geometry)


class VisualizerDiffEqPiecewiseConstant(Visualizer):
    def __init__(self, sm: SolutionsManager, diffusion_contrast_lower, diffusion_contrast_upper, max_vn_dim,
                 num_points_per_dim_to_plot=50, axes_xy_proportions=(3, 3)):
        super().__init__(sm)
        self.diffusion_contrast_lower = diffusion_contrast_lower
        self.diffusion_contrast_upper = diffusion_contrast_upper
        self.max_vn_dim = max_vn_dim
        self.num_points_per_dim_to_plot = num_points_per_dim_to_plot
        self.axes_xy_proportions = axes_xy_proportions

    def __diffusion_coefficients_slider(self,
                                        layout={'height': '450px', 'width': '750px', 'justify-content': 'justify'}):
        grid = GridspecLayout(self.sm.blocks_geometry[0] + 1, self.sm.blocks_geometry[1], layout=layout)
        cells = list(itertools.product(*list(map(range, self.sm.blocks_geometry))))
        coefs_sliders = dict()
        for i, j in cells:
            key = f"a{i}{j}"
            coefs_sliders[key] = widgets.IntSlider(
                value=int(((116 * i) ** 2 + 737 * j + 42) % (
                        self.diffusion_contrast_upper - self.diffusion_contrast_lower) + self.diffusion_contrast_lower),
                min=self.diffusion_contrast_lower, max=self.diffusion_contrast_upper,
                step=1,
                description=f'a[{i},{j}]:', disabled=False,
                continuous_update=False, orientation='horizontal',
                readout=True)
            grid[i, j] = coefs_sliders[key]
        return grid, coefs_sliders

    def show_diffusion_coefficient_subdomains(self, axes_xy_proportions=(7, 7)):
        """

        :param axes_xy_proportions: Overrides self.axes_xy_proportions if it is not None. By default it does.
        :return:
        """
        axes_xy_proportions = self.axes_xy_proportions if axes_xy_proportions is None else axes_xy_proportions
        grid, coefs_sliders = self.__diffusion_coefficients_slider()

        def show_diffusion(**diffusion_coefficients):
            output = widgets.Output(layout={'justify-content': 'center'})  # 'width': '100%', 'height': '600px',
            with output:
                fig, ax = plt.subplots(1, 2, figsize=np.array(axes_xy_proportions) * np.array([1, 2]))

                # sm = SolutionsManagerFEM(blocks_geometry=self.sm.blocks_geometry, discretization=10, num_cores=1)
                diffusion_coefficients = self.reshape_diffusion_coefficients_to_viz(diffusion_coefficients)
                u = self.sm.generate_solutions(diffusion_coefficients)
                num_points_per_dim_to_plot = np.prod(self.sm.blocks_geometry) * 10
                x, y, u_reshaped = prepare_solution4plot(self.sm, u, num_points_per_dim_to_plot)
                u_reshaped = u_reshaped.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot))
                plot_solution(ax[1], x, y, u_reshaped, self.sm, contour_levels=0, vmin=None, vmax=None, colorbar=False,
                              cmap=SOLUTION_CMAP, add_grid=True)
                ax[1].set_yticks([])

                sns.heatmap(np.squeeze(diffusion_coefficients[:, ::-1]), vmin=self.diffusion_contrast_lower - 1,
                            vmax=self.diffusion_contrast_upper + 1,
                            annot=True, ax=ax[0], linestyle="dashed", square=True,
                            xticklabels=False, yticklabels=False, linewidths=1, cbar=False, cmap="Greys")
                plt.tight_layout()

                plt.show()
            grid[-1, :] = output

        out = widgets.interactive_output(show_diffusion, coefs_sliders)
        display(grid, out)

    def show_forward_problem_approximations(self, reduced_basis_dict):
        def show_approx(n_dim, rb_methods, **diffusion_coefficients):
            approximate_solutions = []
            diffusion_coefficients = self.reshape_diffusion_coefficients_to_viz(diffusion_coefficients)
            solution = self.sm.generate_solutions(diffusion_coefficients)
            for rb_method in rb_methods:
                rb = reduced_basis_dict[rb_method][:n_dim]
                approximate_solutions.append(self.sm.generate_fm_solutions(theta=diffusion_coefficients, basis=rb))

            plot_solutions_together(
                self.sm,
                None, [solution] + approximate_solutions,
                num_points_per_dim_to_plot=self.num_points_per_dim_to_plot,
                contour_levels=7,
                axes_xy_proportions=self.axes_xy_proportions,
                titles=["True solution"] + list(rb_methods), colorbar=False,
                measurement_points=None)
            plt.show()

        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(2, 2)
        global_grid[0, :], available_widgets = self.__diffusion_coefficients_slider(
            layout={'justify-content': 'justify'})
        global_grid[1, 0] = available_widgets["rb_methods"] = widgets.SelectMultiple(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys()),
            description="Reduced Basis: ",
            disabled=False,
            style=style)
        global_grid[1, 1] = available_widgets["n_dim"] = widgets.IntSlider(
            value=1, min=1,
            max=50,
            step=1,
            description='RB dim n:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_approx, available_widgets)
        display(global_grid, out)

    def show_forward_problem_non_linear_approximations(self, thetas2train, reduced_basis_dict, sklearn_models,
                                                       offline_fit):
        def show_approx(n_linear, n_non_linear, rb_method, sklearn_model, **diffusion_coefficients):
            diffusion_coefficients = self.reshape_diffusion_coefficients_to_viz(diffusion_coefficients)
            solution = self.sm.generate_solutions(diffusion_coefficients)
            rb = reduced_basis_dict[rb_method]
            online_predictor = offline_fit(thetas2train, rb, n_linear, n_non_linear, sklearn_models[sklearn_model])
            approximate_solutions = [
                self.sm.generate_fm_solutions(theta=diffusion_coefficients, basis=rb[:n_linear]),
                online_predictor(diffusion_coefficients),
                self.sm.generate_fm_solutions(theta=diffusion_coefficients, basis=rb[:n_linear + n_non_linear]),
            ]

            plot_solutions_together(
                self.sm,
                None, [solution] + approximate_solutions,
                num_points_per_dim_to_plot=self.num_points_per_dim_to_plot,
                contour_levels=7,
                axes_xy_proportions=self.axes_xy_proportions,
                titles=["True solution"] + [f"{rb_method} dim {n_linear}", "Non linear",
                                            f"{rb_method} dim {n_linear + n_non_linear}"], colorbar=False,
                measurement_points=None)
            plt.show()

        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(3, 2)
        global_grid[0, :], available_widgets = self.__diffusion_coefficients_slider(
            layout={'justify-content': 'justify'})
        global_grid[1, 0] = available_widgets["rb_method"] = widgets.Select(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys())[0],
            description="Reduced Basis: ",
            disabled=False,
            style=style)
        global_grid[1, 1] = available_widgets["sklearn_model"] = widgets.Select(
            options=list(sklearn_models.keys()),
            value=list(sklearn_models.keys())[0],
            description="Non linear models: ",
            disabled=False,
            style=style)
        global_grid[2, 0] = available_widgets["n_linear"] = widgets.IntSlider(
            value=2, min=1,
            max=self.max_vn_dim,
            step=1,
            description='RB dim n:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)
        global_grid[2, 1] = available_widgets["n_non_linear"] = widgets.IntSlider(
            value=1, min=1,
            max=self.max_vn_dim,
            step=1,
            description='non linear dim n:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_approx, available_widgets)
        display(global_grid, out)

    def show_forward_problem_convergence(self, theta, solutions, reduced_basis_dict, max_vn_dim=None):
        max_vn_dim = self.max_vn_dim if max_vn_dim is None else max_vn_dim

        def show_convergence(rb_methods, error_metric):
            for rb_method in rb_methods:
                errors = []
                for n in range(1, max_vn_dim):
                    basis = reduced_basis_dict[rb_method][:n]
                    approximate_solutions = self.sm.generate_fm_solutions(theta=theta, basis=basis)
                    errors.append(error_metrics_dict[error_metric](self.sm, solutions - approximate_solutions))

                plt.plot(np.arange(1, max_vn_dim, dtype=int), errors, ".-", label=rb_method)
            plt.xticks(np.arange(1, max_vn_dim, dtype=int))
            plt.yscale("log")
            plt.grid()
            plt.legend()
            plt.show()

        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(1, 2)
        available_widgets = dict()

        global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
            options=list(error_metrics_dict.keys()),
            description="Error metric: ",
            disabled=False,
            style=style)

        global_grid[0, 1] = available_widgets["rb_methods"] = widgets.SelectMultiple(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys()),
            description="Reduced Basis: ",
            disabled=False,
            style=style)

        out = widgets.interactive_output(show_convergence, available_widgets)
        display(global_grid, out)

    def show_forward_problem_non_linear_convergence(self, theta, solutions, reduced_basis_dict, sklearn_models,
                                                    offline_fit, max_vn_dim=None):
        max_vn_dim = self.max_vn_dim if max_vn_dim is None else max_vn_dim

        def show_convergence(n_non_linear, rb_method, sklearn_model, error_metric):
            rb = reduced_basis_dict[rb_method]
            errors = []
            for n_linear in tqdm(range(1, max_vn_dim - n_non_linear), desc="Calculating convergence errors."):
                online_predictor = offline_fit(theta, rb, n_linear, n_non_linear, sklearn_models[sklearn_model])
                approximate_solutions = [
                    self.sm.generate_fm_solutions(theta=theta, basis=rb[:n_linear]),
                    online_predictor(theta),
                    self.sm.generate_fm_solutions(theta=theta, basis=rb[:n_linear + n_non_linear]),
                ]

                errors.append(
                    [error_metrics_dict[error_metric](self.sm, solutions - approx) for approx in approximate_solutions])

            for error, name in zip(np.transpose(errors),
                                   ["Linear method", "Non-linear method", "Optimal Non-linear method"]):
                plt.plot(np.arange(1, max_vn_dim - n_non_linear, dtype=int), error, ".-", label=name)
            plt.xticks(np.arange(1, max_vn_dim - n_non_linear, dtype=int))
            plt.yscale("log")
            plt.grid()
            plt.legend()
            plt.show()

        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(2, 2)
        available_widgets = dict()
        style = {'description_width': 'initial'}
        global_grid[0, 0] = available_widgets["rb_method"] = widgets.Select(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys())[0],
            description="Reduced Basis: ",
            disabled=False,
            style=style)
        global_grid[0, 1] = available_widgets["sklearn_model"] = widgets.Select(
            options=list(sklearn_models.keys()),
            value=list(sklearn_models.keys())[0],
            description="Non linear models: ",
            disabled=False,
            style=style)
        global_grid[1, 0] = available_widgets["error_metric"] = widgets.Dropdown(
            options=list(error_metrics_dict.keys()),
            description="Error metric: ",
            disabled=False,
            style=style)
        global_grid[1, 1] = available_widgets["n_non_linear"] = widgets.IntSlider(
            value=1, min=1,
            max=self.max_vn_dim,
            step=1,
            description='non linear dim n:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_convergence, available_widgets)
        display(global_grid, out)

    # =============== Inverse Problem =============== #
    def get_measurements_and_riesz(self, solutions, m, sampling_method, sigma, calculate_riesz, measurements_pointwise,
                                   measurements_riesz, noise):
        # in the case of optimal sampling we need the basis rb: so it should go inside the loop
        measurement_points = sampling_method(m, self.sm.x_domain, self.sm.y_domain, sm=self.sm)
        if sigma == 0:
            riesz = self.sm.evaluate_solutions(points=measurement_points,
                                               solutions=np.eye(self.sm.vspace_dim))  # (N, m)
            # riesz = calculate_riesz(measurement_points=measurement_points, sm=self.sm, sigma=0.01)
            measurements_function = partial(measurements_pointwise, measurement_points=measurement_points)
        else:
            riesz = calculate_riesz(measurement_points=measurement_points, sm=self.sm, sigma=sigma)
            measurements_function = partial(measurements_riesz, riesz=riesz)
        measurements_online = measurements_function(solutions) + np.random.normal(scale=noise,
                                                                                  size=(len(solutions), m))
        return measurements_function, measurements_online, riesz, measurement_points

    def show_inverse_problem_approximations(self, measurements_sampling_method_dict, reduced_basis_dict,
                                            state_estimation_method_dict, calculate_riesz, measurements_pointwise,
                                            measurements_riesz, add_grid=True, plot_measurement_points_in=[0]):

        def show_approx(n_dim, rb_methods, m, measurements_sampling_method, state_estimation_method,
                        sigma, **diffusion_coefficients):
            diffusion_coefficients = self.reshape_diffusion_coefficients_to_viz(diffusion_coefficients)
            solution = self.sm.generate_solutions(diffusion_coefficients)
            measurements_function, measurements_online, riesz, measurement_points = (
                self.get_measurements_and_riesz(solution, m,
                                                measurements_sampling_method_dict[measurements_sampling_method],
                                                sigma, calculate_riesz, measurements_pointwise, measurements_riesz,
                                                noise=0))
            approximate_solutions = []
            for rb_method in rb_methods:
                rb = reduced_basis_dict[rb_method][:n_dim]
                approximate_solutions.append(
                    state_estimation_method_dict[state_estimation_method](measurements_function, measurements_online,
                                                                          rb, riesz=riesz))

            plot_solutions_together(
                self.sm,
                None, [solution] + approximate_solutions,
                num_points_per_dim_to_plot=self.num_points_per_dim_to_plot,
                contour_levels=7, add_grid=add_grid,
                axes_xy_proportions=self.axes_xy_proportions,
                titles=["True solution"] + list(rb_methods), colorbar=False,
                measurements_color="white", measurements_edge_color="black",
                measurement_points=measurement_points, plot_measurement_points_in=plot_measurement_points_in)
            plt.show()

        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(4, 2)
        global_grid[0, :], available_widgets = self.__diffusion_coefficients_slider(
            layout={'justify-content': 'justify'})
        global_grid[1, 0] = available_widgets["rb_methods"] = widgets.SelectMultiple(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys()),
            description="Reduced Basis: ",
            disabled=False,
            style=style)
        global_grid[1, 1] = available_widgets["n_dim"] = widgets.IntSlider(
            value=1, min=1,
            max=50,
            step=1,
            description='RB dim n:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
            options=list(measurements_sampling_method_dict.keys()),
            description="Measurements sampling method: ",
            disabled=False,
            style=style)
        global_grid[2, 1] = available_widgets["m"] = widgets.IntSlider(
            value=50, min=self.max_vn_dim,
            max=10 * self.max_vn_dim,
            step=1,
            description='Number of measurements:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)
        global_grid[3, 0] = available_widgets["state_estimation_method"] = widgets.Dropdown(
            options=list(state_estimation_method_dict.keys()),
            description="State estimation method: ",
            disabled=False,
            style=style)

        global_grid[3, 1] = available_widgets["sigma"] = widgets.FloatSlider(
            value=0, min=0,
            max=2,
            step=0.01,
            description='Sensor width: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_approx, available_widgets)
        display(global_grid, out)

    def show_inverse_problem_convergence(self, solutions, measurements_sampling_method_dict, reduced_basis_dict,
                                         state_estimation_method_dict, calculate_riesz, measurements_pointwise,
                                         measurements_riesz):
        def show_convergence(rb_methods, measurements_sampling_method, m, state_estimation_method, error_metric, noise,
                             sigma):
            measurements_function, measurements_online, riesz, measurement_points = (
                self.get_measurements_and_riesz(solutions, m,
                                                measurements_sampling_method_dict[measurements_sampling_method],
                                                sigma, calculate_riesz, measurements_pointwise, measurements_riesz,
                                                noise=0))
            vn_space_dim = list(range(1, self.max_vn_dim))
            errors = defaultdict(list)
            for rb_method, n in tqdm(itertools.product(rb_methods, vn_space_dim)):
                basis = np.reshape(reduced_basis_dict[rb_method][:n], (n, -1))
                approximate_solutions = (state_estimation_method_dict[state_estimation_method]
                                         (measurements_function, measurements_online, basis, riesz=riesz))
                errors[rb_method].append(error_metrics_dict[error_metric](self.sm, solutions - approximate_solutions))

            clear_output(wait=True)
            for rb_method, error in errors.items():
                plt.plot(vn_space_dim, error, ".-", label=rb_method)
            plt.xticks(np.arange(1, self.max_vn_dim, dtype=int))
            plt.yscale("log")
            plt.grid()
            plt.legend()
            plt.show()

        # ----------------- Display plot style ----------------- #
        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(4, 2)
        available_widgets = dict()

        global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
            options=list(error_metrics_dict.keys()),
            description="Error metric: ",
            disabled=False,
            style=style)
        global_grid[0, 1] = available_widgets["noise"] = widgets.FloatSlider(
            value=0,
            min=0,
            max=1,
            step=0.01,
            description="Noise: ",
            disabled=False,
            style=style)

        global_grid[1, :] = available_widgets["rb_methods"] = widgets.SelectMultiple(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys()),
            description="Reduced Basis: ",
            disabled=False,
            style=style)

        global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
            options=list(measurements_sampling_method_dict.keys()),
            description="Measurements sampling method: ",
            disabled=False,
            style=style)
        global_grid[2, 1] = available_widgets["m"] = widgets.IntSlider(
            value=50, min=self.max_vn_dim,
            max=10 * self.max_vn_dim,
            step=1,
            description='Number of measurements:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)
        global_grid[3, 0] = available_widgets["state_estimation_method"] = widgets.Dropdown(
            options=list(state_estimation_method_dict.keys()),
            description="State estimation method: ",
            disabled=False,
            style=style)

        global_grid[3, 1] = available_widgets["sigma"] = widgets.FloatSlider(
            value=0, min=0,
            max=2,
            step=0.01,
            description='Sensor width: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_convergence, available_widgets)
        display(global_grid, out)

    def compare_inverse_problem_state_estimation_methods(self, solutions, measurements_sampling_method_dict,
                                                         reduced_basis_dict, state_estimation_method_dict,
                                                         calculate_riesz, measurements_pointwise, measurements_riesz):
        def show_semethod(rb_method, measurements_sampling_method, m, state_estimation_methods, error_metric, noise,
                          vn_range, sigma):
            measurements_function, measurements_online, riesz, measurement_points = (
                self.get_measurements_and_riesz(solutions, m,
                                                measurements_sampling_method_dict[measurements_sampling_method],
                                                sigma, calculate_riesz, measurements_pointwise, measurements_riesz,
                                                noise=0))
            vn_space_dim = list(range(*vn_range))
            errors = defaultdict(list)
            for state_estimation_method, n in tqdm(itertools.product(state_estimation_methods, vn_space_dim)):
                basis = np.reshape(reduced_basis_dict[rb_method][:n], (n, -1))
                approximate_solutions = (state_estimation_method_dict[state_estimation_method]
                                         (measurements_function, measurements_online, basis, riesz=riesz))
                errors[state_estimation_method].append(
                    error_metrics_dict[error_metric](self.sm, solutions - approximate_solutions))

            clear_output(wait=True)
            for state_estimation_method, error in errors.items():
                plt.plot(vn_space_dim, error, ".-", label=state_estimation_method)
            plt.xticks(vn_space_dim)
            plt.grid()
            plt.yscale("log")
            plt.ylim((None, 1e-1))
            plt.legend()
            plt.show()

        # ----------------- Display plot style ----------------- #
        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(5, 2)
        available_widgets = dict()

        global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
            options=list(error_metrics_dict.keys()),
            description="Error metric: ",
            disabled=False,
            style=style)
        global_grid[0, 1] = available_widgets["noise"] = widgets.FloatText(
            value=0,
            min=0,
            max=1,
            # step=0.01,
            description="Noise: ",
            disabled=False,
            style=style)

        global_grid[1, :] = available_widgets["rb_method"] = widgets.Dropdown(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys())[0],
            description="Reduced Basis: ",
            disabled=False,
            style=style)

        global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
            options=list(measurements_sampling_method_dict.keys()),
            # value=list(measurements_sampling_method_dict.keys()),
            description="Measurements sampling method: ",
            disabled=False,
            style=style)

        global_grid[2, 1] = available_widgets["m"] = widgets.IntText(
            value=50, min=self.max_vn_dim,
            # max=10 * max_vn_dim,
            # step=1,
            description='Number of measurements:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)
        global_grid[3, 0] = available_widgets["state_estimation_methods"] = widgets.SelectMultiple(
            options=list(state_estimation_method_dict.keys()),
            value=list(state_estimation_method_dict.keys()),
            description="State estimation method: ",
            disabled=False,
            style=style)

        global_grid[3, 1] = available_widgets["vn_range"] = widgets.IntRangeSlider(
            min=0,
            max=self.max_vn_dim,
            value=(1, self.max_vn_dim),
            step=1,
            description="dim(Vn) range: ",
            disabled=False,
            style=style)

        global_grid[4, :] = available_widgets["sigma"] = widgets.FloatSlider(
            value=0, min=0,
            max=2,
            step=0.01,
            description='Sensor width: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_semethod, available_widgets)
        display(global_grid, out)

    def compare_inverse_problem_samplers(self, solutions, measurements_sampling_method_dict, reduced_basis_dict,
                                         state_estimation_method_dict, calculate_riesz, measurements_pointwise,
                                         measurements_riesz):
        def show_smethod(rb_method, measurements_sampling_methods, m, state_estimation_method, error_metric, noise,
                         vn_range, sigma):

            vn_space_dim = list(range(*vn_range))
            errors = defaultdict(list)
            for measurements_sampling_method, n in tqdm(itertools.product(measurements_sampling_methods, vn_space_dim)):
                basis = np.reshape(reduced_basis_dict[rb_method][:n], (n, -1))
                measurements_function, measurements_online, riesz, measurement_points = (
                    self.get_measurements_and_riesz(solutions, m,
                                                    measurements_sampling_method_dict[measurements_sampling_method],
                                                    sigma, calculate_riesz, measurements_pointwise,
                                                    measurements_riesz, noise=noise))
                approximate_solutions = (state_estimation_method_dict[state_estimation_method]
                                         (measurements_function, measurements_online, basis, riesz=riesz))
                errors[measurements_sampling_method].append(
                    error_metrics_dict[error_metric](self.sm, solutions - approximate_solutions))

            clear_output(wait=True)
            for measurements_sampling_method, error in errors.items():
                plt.plot(vn_space_dim, error, ".-", label=measurements_sampling_method)

            plt.xticks(np.arange(*vn_range, dtype=int))
            plt.grid()
            plt.yscale("log")
            plt.ylim((None, 1e-1))
            plt.legend()
            plt.show()

        style = {'description_width': 'initial'}
        global_grid = GridspecLayout(5, 2)
        available_widgets = dict()

        global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
            options=list(error_metrics_dict.keys()),
            description="Error metric: ",
            disabled=False,
            style=style)
        global_grid[0, 1] = available_widgets["noise"] = widgets.FloatText(
            value=0,
            min=0,
            max=1,
            # step=0.01,
            description="Noise: ",
            disabled=False,
            style=style)

        global_grid[1, :] = available_widgets["rb_method"] = widgets.Dropdown(
            options=list(reduced_basis_dict.keys()),
            value=list(reduced_basis_dict.keys())[0],
            description="Reduced Basis: ",
            disabled=False,
            style=style)

        global_grid[2, 0] = available_widgets["measurements_sampling_methods"] = widgets.SelectMultiple(
            options=list(measurements_sampling_method_dict.keys()),
            value=list(measurements_sampling_method_dict.keys()),
            description="Measurements sampling method: ",
            disabled=False,
            style=style)

        global_grid[2, 1] = available_widgets["m"] = widgets.IntText(
            value=50, min=self.max_vn_dim,
            # max=10 * max_vn_dim,
            # step=1,
            description='Number of measurements:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)
        global_grid[3, 0] = available_widgets["state_estimation_method"] = widgets.Dropdown(
            options=list(state_estimation_method_dict.keys()),
            # value=list(state_estimation_method_dict.keys()),
            description="State estimation method: ",
            disabled=False,
            style=style)

        global_grid[3, 1] = available_widgets["vn_range"] = widgets.IntRangeSlider(
            min=0,
            max=self.max_vn_dim,
            value=(1, self.max_vn_dim),
            step=1,
            description="dim(Vn) range: ",
            disabled=False,
            style=style)

        global_grid[4, :] = available_widgets["sigma"] = widgets.FloatSlider(
            value=0, min=0,
            max=2,
            step=0.01,
            description='Sensor width: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            style=style)

        out = widgets.interactive_output(show_smethod, available_widgets)
        display(global_grid, out)

    def visualize_riezs_representers(self, sm, calculate_gauss_riesz):
        @widgets.interact(point_x=widgets.FloatSlider(min=sm.x_domain[0], max=sm.x_domain[1], value=sm.x_domain[1] / 2),
                          point_y=widgets.FloatSlider(min=sm.y_domain[0], max=sm.y_domain[1], value=sm.y_domain[1] / 2),
                          sigma=widgets.FloatSlider(min=0.01, max=1, value=0.5),
                          difference=False,
                          )
        def plot_riesz_representers(point_x, point_y, sigma, difference):
            measurement_points = np.array([[point_x, point_y]])
            riesz = calculate_gauss_riesz(sm, measurement_points, sigma)  # (N, m)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            # plot riesz representer
            x, y, u = prepare_solution4plot(sm, riesz, self.num_points_per_dim_to_plot)
            u = u.reshape((self.num_points_per_dim_to_plot, self.num_points_per_dim_to_plot))
            plot_solution(ax[0], x, y, u, sm, contour_levels=0, vmin=None, vmax=None,
                          colorbar=False, cmap=SOLUTION_CMAP, add_grid=True)
            ax[0].set_title("Riesz representer")

            # plot original gaussian or difference
            x, y = np.meshgrid(np.linspace(*sm.x_domain, num=self.num_points_per_dim_to_plot),
                               np.linspace(*sm.y_domain, num=self.num_points_per_dim_to_plot))
            points = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
            li = np.exp(-cdist(measurement_points, points) ** 2 / 2 / sigma ** 2).reshape(
                (self.num_points_per_dim_to_plot, self.num_points_per_dim_to_plot))
            plot_solution(ax[1], x, y, li - u if difference else li, sm, contour_levels=0, vmin=None, vmax=None,
                          colorbar=False, cmap=SOLUTION_CMAP, add_grid=True)
            ax[1].set_title(
                "Differnece between projected and original gaussian" if difference else "Not projected gaussian")

            # plot point
            ax[0].scatter(point_x, point_y, c="white", s=10)
            ax[1].scatter(point_x, point_y, c="white", s=10)
