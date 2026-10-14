from src.config import figures_path
from src.lib.SolutionsManagers import SolutionsManagerFEM
from src.lib.VizUtils import save_fig
from src.notebooks.functions4notebooks import *

sm = SolutionsManagerFEM(blocks_geometry=(2, 2), discretization=15, num_cores=1)
diffusion_contrast_lower = 1
diffusion_contrast_upper = 10
diffusion_coefficients = [1, 1, 2, 3]
diffusion_coefficients = reshape_diffusion_coefficients_to_viz(diffusion_coefficients, sm.blocks_geometry)
particular_solution_of_poisson = sm.generate_solutions(diffusion_coefficients)

if __name__ == "__main__":
    plt.style.use("figures.mplstyle")
    axes_xy_proportions = (6, 6)
    np.random.seed(42)

    with save_fig(f"{figures_path}/poisson_example.pdf", dpi=None):
        fig, ax = plt.subplots(1, 2, figsize=np.array(axes_xy_proportions) * np.array([1, 0.5]))
        num_points_per_dim_to_plot = np.prod(sm.blocks_geometry) * 10
        x, y, u_reshaped = prepare_solution4plot(sm, particular_solution_of_poisson, num_points_per_dim_to_plot)
        u_reshaped = u_reshaped.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot))
        plot_solution(ax[1], x, y, u_reshaped, sm, contour_levels=0, vmin=None, vmax=None, colorbar=False,
                      cmap=SOLUTION_CMAP, add_grid=True)
        ax[1].set_yticks([])
        ax[1].set_xticks([])

        sns.heatmap(np.squeeze(diffusion_coefficients[:, ::-1]), vmin=diffusion_contrast_lower - 1,
                    vmax=diffusion_contrast_upper + 1,
                    annot=True, ax=ax[0], linestyle="dashed", square=True,
                    xticklabels=False, yticklabels=False, linewidths=1, cbar=False, cmap="Greys")
        fig.tight_layout()
