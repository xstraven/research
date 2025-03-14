import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad

key = jax.random.key(0)


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)


# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Build a toy dataset.
inputs = jnp.array(
    [
        [0.52, 1.12, 0.77],
        [0.88, -1.08, 0.15],
        [0.52, 0.06, -1.30],
        [0.74, -2.49, 1.39],
    ]
)
targets = jnp.array([True, True, False, True])


# Training loss is the negative log-likelihood of the training examples.
def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))


# Initialize random model coefficients
key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (3,))
b = jax.random.normal(b_key, ())
