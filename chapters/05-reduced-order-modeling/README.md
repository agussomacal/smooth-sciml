# Chapter 5: Reduced order modeling and inverse state estimation

**Authors:** Olga Mula and Agustín Somacal

This folder accompanies the chapter on reduced order modeling of parametric
PDEs and applications to inverse state estimation.

## 🧭 Guide into the code

Two main notebooks (found in `/src/notebooks/`) gather all the experiments:

- `LinearRom.ipynb`
- `NonLinearRom.ipynb`

In `LinearRom.ipynb` you will find numerical experiments to help understand linear reduced order modeling using
the diffusion equation as guiding example. There you will find:

1. The problem setting.
2. Different strategies of how to build a reduced basis (RB).
3. How to accelerate the solution of the forward problem using RB.
4. To explore how the Riesz representers look like.
5. How to implement PBDW.
6. How to solve inverse problems with RB and PBDW.

In `NonLinearRom.ipynb` you will find numerical experiments to understand the need and the effect of using non-linear
spaces to approximate a given manifold. In this case the guiding example is the step function.

## ⏳ Setting up environments to locally run examples

Requires **Python >= 3.12** to work with **jax>=0.5.0**.

Create virtual environment, here are some alternative ways

### Using `uv`
 

```
uv venv --python 3.12
```

Install libraries (`uv` looks at `.toml` file to find and solve dependencies)

```
uv sync
```

### Using `venv` or `virtualenv`

```
python -m venv .venv
virtualenv .venv
```

Activate virtual environment

```
. .venv/bin/activate
```

Installing libraries with virtual environments created with `venv` or `virtualenv` use the `requirements.txt` file to get the dependencies:

```
pip install -r requirements.txt
```

In order to be able to run jupyter notebooks with this virtual environment's kernel:

```
python -m ipykernel install --user --name .venv --display-name "mlbook"
```

