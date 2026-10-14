import itertools
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

N_train_max = 10000


@jax.jit
# @partial(jax.vmap, in_axes=(0, 0, 0, 0))
def cos_coeff(c, h, w, k):
    return 2 * h * jnp.sin(jnp.pi * k * w) * jnp.cos(jnp.pi * k * (2 * c + w)) / jnp.pi / k


@jax.jit
# @partial(jax.vmap, in_axes=(0, 0, 0, 0))
def sin_coeff(c, h, w, k):
    return 2 * h * jnp.sin(jnp.pi * k * w) * jnp.sin(jnp.pi * k * (2 * c + w)) / jnp.pi / k


def step_function_fourier_coeffs(c, h, w, k_max):
    ccoeff0 = jnp.array([h * w * 2])
    ccoeffk = cos_coeff(c, h, w, jnp.arange(1, k_max))
    scoeffk = sin_coeff(c, h, w, jnp.arange(1, k_max))
    return jnp.concat((ccoeff0, ccoeffk, scoeffk))
    # return jnp.array((ccoeff0,) + tuple(itertools.chain(*zip(scoeffk, ccoeffk))))


def order_coeffs_by_freq(coeffs):
    k_max = (len(coeffs) - 1) // 2
    return jnp.array((coeffs[0],) + tuple(itertools.chain(*zip(coeffs[1:k_max + 1], coeffs[k_max + 1:]))))


def disorder_coeffs_by_freq(coeffs):
    k_max = (len(coeffs) - 1) // 2
    return jnp.concat((coeffs[:1], coeffs[1::2], coeffs[2::2]))


@jax.jit
# @partial(jax.vmap, in_axes=(None, 1))
def inv_ft(x, coeffs):
    f = coeffs[0] / 2 + 0 * x
    k_max = (len(coeffs) - 1) // 2
    for i in range(k_max):
        k = i + 1
        f += coeffs[1 + i + k_max] * jnp.sin(2 * jnp.pi * k * x) + coeffs[1 + i] * jnp.cos(2 * jnp.pi * k * x)
    return f


@jax.jit
def modulate_coeffs(coeffs, modulation):
    k_max = len(modulation) - 1
    # return jnp.concat((jnp.array([jnp.min(jnp.array((coeffs[:1], modulation[:1])))]),
    #                    jnp.min(jnp.array((coeffs[1:k_max+1], modulation[1:])), axis=1),
    #                    jnp.min(jnp.array((coeffs[k_max+1:], modulation[1:])), axis=1)))
    return jnp.concat(
        (coeffs[:1] * modulation[0], coeffs[1:k_max + 1] * modulation[1:], coeffs[k_max + 1:] * modulation[1:]))


def power_modulation(coeffs, power):
    k_max = (len(coeffs) - 1) // 2
    energy = jnp.sum(coeffs ** 2)
    k = jnp.arange(k_max + 1)
    default_modulation = 1 / (k + 1)
    modulation = 1 / (k + 1) ** power / default_modulation
    coeffs_modulated = modulate_coeffs(coeffs, modulation)
    coeffs_modulated *= jnp.sqrt(energy / jnp.sum(coeffs_modulated ** 2))
    return coeffs_modulated


def sample_space_for_non_linear_train(n_train, n, n_non_linear, k_max, power=0, c_lim=(0, 0, 1), h_lim=(1., 1., 1),
                                      w_lim=(0., 1., N_train_max), seed=42):
    np.random.seed(seed)
    c = np.linspace(*c_lim)
    h = np.linspace(*h_lim)
    w = np.linspace(*w_lim)
    input_params = np.array(list(itertools.product(c, h, w)))[np.random.choice(n_train, size=n_train, replace=False)]
    coeffs = jax.vmap(partial(step_function_fourier_coeffs, k_max=k_max), (0, 0, 0))(*input_params.T)
    if power > 0: coeffs = jax.vmap(power_modulation, (0, None))(coeffs, power)
    ordered_coeffs = jax.vmap(order_coeffs_by_freq, (0,))(coeffs)
    coeffs_known = ordered_coeffs[:, :n]
    coeffs_to_learn = ordered_coeffs[:, n:n + n_non_linear]
    return coeffs_known, coeffs_to_learn


# ============================================================= #
#                           TEST
# ============================================================= #
if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor

    k_max = 100  # maximum number of Fourier terms for the approximations
    Nx = 500  # number of discretization points in space
    N_train_max = 10000
    n_train = N_train_max  # number of examples to train

    # Defines the parameter ranges of \theta, here is set to only vary the width being always of height=1 and left starting point in 0
    c_lim = (0.0, 0.0, 1)
    h_lim = (1.0, 1.0, 1)
    w_lim = (0.0, 1.0, N_train_max)

    smoothness = 1
    n_linear = 3
    n_to_learn = 1
    power = 1.
    n = 3
    n_non_linear = 1
    n_train = 100
    log_x = True
    log_y = True

    coeffs_known, coeffs_to_learn = sample_space_for_non_linear_train(n_train, n, n_non_linear, k_max, power=0,
                                                                      c_lim=(0, 0, 1), h_lim=(1., 1., 1),
                                                                      w_lim=(0., 1., N_train_max))

    assert np.sum(np.isnan(coeffs_known)) == 0
    assert np.sum(np.isnan(coeffs_to_learn)) == 0
    model = RandomForestRegressor(n_estimators=10)
    model.fit(np.array(coeffs_known), np.array(coeffs_to_learn))
    error = (np.array(coeffs_to_learn) - model.predict(np.array(coeffs_known)).reshape(coeffs_to_learn.shape)) ** 2
