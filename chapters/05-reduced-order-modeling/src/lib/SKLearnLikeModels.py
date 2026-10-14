from functools import partial
from typing import Tuple

import jax
import optax
from flax import nnx
from jax import numpy as jnp


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        return x @ self.w + self.b


class MLP(nnx.Module):
    def __init__(self, din: int, layers: Tuple[int, ...], dout: int, *, rngs: nnx.Rngs, activation=nnx.sigmoid):
        self.layers = [Linear(iin, iout, rngs=rngs) for iin, iout in zip((din,) + layers, layers + (dout,))]
        self.activation = activation

    def __call__(self, x: jax.Array):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = self.activation(x)
            x = layer(x)
        return x


class SKJaxMLP:
    def __init__(self, layers: Tuple[int, ...], activation=nnx.sigmoid, num_iters=1000, lr=1e-3):
        self.model_partial = partial(MLP, layers=layers, rngs=nnx.Rngs(0), activation=activation)
        self.model = None
        self.num_iters = num_iters
        self.optimizer = optax.adam(lr)

    def fit(self, X, y):
        X = jnp.array(X)
        y = jnp.array(y)

        self.decay = jnp.std(y, axis=0)
        # k_modes = list(range((y.shape[1]-1)//2))
        # decay = np.append([1], np.ravel(np.array([k_modes, k_modes]), order="F"))

        # self.model = MLP(din=jnp.shape(X)[1], layers=self.layers, dout=jnp.shape(y)[1], rngs=nnx.Rngs(0), activation=self.activation)
        self.model = self.model_partial(din=jnp.shape(X)[1], dout=jnp.shape(y)[1])
        optimizer = nnx.Optimizer(self.model, self.optimizer)  # reference sharing

        @nnx.jit  # Automatic state management
        def train_step(model, optimizer, x, y):
            def loss_fn(model: MLP):
                y_pred = model(x) * self.decay
                return jnp.mean((y_pred - y) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)  # In place updates.

            return loss

        for i in range(self.num_iters):
            train_step(self.model, optimizer, X, y)
            if i % 100 == 0: print(f"\rTraining {i} / {self.num_iters}", end="")

    def predict(self, X):
        return self.model(X) * self.decay
