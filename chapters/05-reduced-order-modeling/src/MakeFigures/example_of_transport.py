import matplotlib
import numpy as np
from flax import nnx
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.config import figures_path, data_temp_path
from src.lib.SKLearnLikeModels import SKJaxMLP
from src.lib.Utils4TransportEq import step_function_fourier_coeffs, power_modulation, order_coeffs_by_freq, \
    disorder_coeffs_by_freq, inv_ft, sample_space_for_non_linear_train, N_train_max
import jax.numpy as jnp
from perplexitylab.miscellaneous import if_exist_load_else_do, make_hash

from src.lib.VizUtils import save_fig

c_lim = (0.0, 0.0, 1)
h_lim = (1.0, 1.0, 1)
w_lim = (0.0, 1.0, N_train_max)

smoothness = 1
n_linear = 10
n_to_learn = 1

TARGET = r"$u(\theta)$"
LINEAR_SVD = "SVD basis"
LINEAR_GREEDY = "Greedy basis"
NCROM_RF = "Nonlinear Compressive ROM via Random Forest"
NCROM_NN = "Nonlinear Compressive ROM via Neural Network"
NCROM_Q = "Nonlinear Compressive ROM via Quadratic model"

non_linear_models = {
    NCROM_RF: RandomForestRegressor(n_estimators=10),
    NCROM_NN: SKJaxMLP(layers=(10, 10), num_iters=10000, lr=1e-2, activation=nnx.sigmoid),
    NCROM_Q: Pipeline([("poly", PolynomialFeatures(degree=2)), ("lasso", MultiTaskLassoCV())])
}


def train(model, n_train, n, n_non_linear, k_max, c_lim, h_lim, w_lim, error=None, hashes=0):
    new_hash = make_hash((n_train, n, n_non_linear, k_max, c_lim, h_lim, w_lim))
    if new_hash != hashes:
        hashes = new_hash
        # ----- create dataset for training non linear models ----- #
        coeffs_known, coeffs_to_learn = (if_exist_load_else_do(check_hash=True)(sample_space_for_non_linear_train)
                                         (path=data_temp_path,
                                          n_train=n_train, n=n, n_non_linear=n_non_linear,
                                          k_max=k_max, power=0,
                                          c_lim=c_lim, h_lim=h_lim, w_lim=w_lim))
        # train
        print("Training model", str(model))
        model.fit(np.array(coeffs_known), np.array(coeffs_to_learn))
        error = (np.array(coeffs_to_learn) - model.predict(np.array(coeffs_known))) ** 2
    return model, error, hashes


def get_reconstruction(c, h, w, n_linear, models, k_max, Nx, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim,
                       n_train=N_train_max, non_linear_models=non_linear_models, errors=None, hashes=None):
    coeffs_predictions = dict()
    reconstruction = dict()
    x = jnp.linspace(0, 1, Nx)

    coeffs_predictions[TARGET] = step_function_fourier_coeffs(c, h, w, k_max=k_max)
    # coeffs_predictions[TARGET] = power_modulation(coeffs, smoothness)
    reconstruction[TARGET] = h * (x < c + w) * (x >= c)

    coeffs_modulated_by_freq = order_coeffs_by_freq(coeffs_predictions[TARGET])
    # ----- linear model svd continuous----- #
    if LINEAR_SVD in models:
        coeffs_modulated_by_freq_filtered = jnp.append(coeffs_modulated_by_freq[:n_linear],
                                                       jnp.zeros(len(coeffs_modulated_by_freq) - n_linear))
        coeffs_predictions[LINEAR_SVD] = disorder_coeffs_by_freq(coeffs_modulated_by_freq_filtered)
        reconstruction[LINEAR_SVD] = inv_ft(x, coeffs_predictions[LINEAR_SVD])

    # ----- greedy model ----- #
    if LINEAR_GREEDY in models:
        dx = (1. / n_linear)
        transition_w = jnp.floor((c + w) / dx) * dx
        proportion_w = ((c + w) - transition_w) / dx
        transition_c = jnp.floor(c / dx) * dx
        proportion_c = (c - transition_c) / dx
        if transition_c == 0: proportion_c = 1  # TODO: other way around to avoid zero in first block?
        reconstruction[LINEAR_GREEDY] = (
                h * (x < c + w) * (x >= c)
                * ((x < transition_w) + (x > transition_w + dx)) * ((x < transition_c) + (x > transition_c + dx))
                + h * proportion_w * (x >= transition_w) * (x <= transition_w + dx)
                + h * proportion_c * (x >= transition_c) * (x <= transition_c + dx)
        )

    # ----- non linear models ----- #
    if errors is None: errors = {model_name: None for model_name in non_linear_models.keys()}
    if hashes is None: hashes = {model_name: 0 for model_name in non_linear_models.keys()}
    for model_name in non_linear_models:
        if model_name in models:
            non_linear_models[model_name], errors[model_name], hashes[model_name] = (
                train(
                    n_train=n_train, model=non_linear_models[model_name], n=n_linear, n_non_linear=k_max - n_linear,
                    k_max=k_max, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim, error=errors[model_name],
                    hashes=hashes[model_name]))
            non_linear_prediction = non_linear_models[model_name].predict(
                np.array(coeffs_modulated_by_freq[:n_linear]).reshape((1, -1))).ravel()
            coeffs_modulated_by_freq_filtered = jnp.concat((coeffs_modulated_by_freq[:n_linear],
                                                            jnp.array(non_linear_prediction), jnp.zeros(
                    len(coeffs_modulated_by_freq) - n_linear - n_to_learn)))
            coeffs_predictions[model_name] = disorder_coeffs_by_freq(coeffs_modulated_by_freq_filtered)
            reconstruction[model_name] = inv_ft(x, coeffs_predictions[model_name])

    # functional_error = dict()
    # for label in reconstruction:
    #     print(label)
    #     if label != TARGET:
    #         functional_error[label] = jnp.sqrt(jnp.mean((reconstruction[TARGET] - reconstruction[label]) ** 2))

    return x, reconstruction, non_linear_models, errors, hashes


def plot_reconstruction(c, h, theta, n_linear, models, k_max, Nx, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim,
                        n_train=N_train_max, non_linear_models=non_linear_models, errors=None, hashes=None, figname="",
                        legend=False):
    colors = matplotlib.colormaps['tab10'].colors
    colors_dict = {
        TARGET: "black",
        LINEAR_SVD: colors[0],
        LINEAR_GREEDY: colors[1],
        NCROM_NN: colors[2],
        NCROM_RF: colors[3],
        NCROM_Q: colors[4]
    }

    x, reconstruction, non_linear_models, errors, hashes = get_reconstruction(
        c=c, h=h, w=theta, n_linear=n_linear, models=models, k_max=k_max, Nx=Nx,
        c_lim=c_lim, h_lim=h_lim, w_lim=w_lim, n_train=n_train,
        non_linear_models=non_linear_models, errors=errors, hashes=hashes)
    # ----- Plotting ----- #
    with save_fig(f"{figures_path}/{figname}transport_eq_reconstructions_Vn{n_linear}.pdf", dpi=None):
        fig, ax = plt.subplots(nrows=1, figsize=(8, 6))
        for label in reconstruction:
            ax.plot(x, reconstruction[label], label=label, c=colors_dict[label])
        ax.set_ylim((-0.25, h * 1.25))
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        if legend:
            down_shift = 0.15
            legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -down_shift))
            legend_height = legend.get_window_extent().height / fig.dpi
            fig.subplots_adjust(bottom=-(legend_height + down_shift)/fig.get_figheight())  # Adjust bottom by the desired amount
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()

    return non_linear_models, errors, hashes


if __name__ == "__main__":
    plt.style.use("figures.mplstyle")

    k_max = 100
    Nx = 500
    models = [
        TARGET,
        LINEAR_SVD,
        LINEAR_GREEDY,
        # NCROM_NN,
        # NCROM_RF
    ]

    non_linear_models, errors, hashes = plot_reconstruction(
        c=0, h=1, theta=0.45, n_linear=3, models=models,
        k_max=k_max, Nx=Nx, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim,
        n_train=N_train_max, non_linear_models=non_linear_models, figname="")
    non_linear_models, errors, hashes = plot_reconstruction(
        c=0, h=1, theta=0.45, n_linear=15, models=models,
        k_max=k_max, Nx=Nx, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim,
        n_train=N_train_max, non_linear_models=non_linear_models, figname="", legend=False)
    non_linear_models, errors, hashes = plot_reconstruction(
        c=0, h=1, theta=0.45, n_linear=30, models=models,
        k_max=k_max, Nx=Nx, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim,
        n_train=N_train_max, non_linear_models=non_linear_models, figname="")

    models = [
        TARGET,
        LINEAR_SVD,
        # LINEAR_GREEDY,
        NCROM_NN,
        # NCROM_RF
    ]
    non_linear_models, errors, hashes = plot_reconstruction(
        c=0, h=1, theta=0.45, n_linear=3, models=models,
        k_max=k_max, Nx=Nx, c_lim=c_lim, h_lim=h_lim, w_lim=w_lim,
        n_train=N_train_max, non_linear_models=non_linear_models, figname="NCROM_", legend=True)
