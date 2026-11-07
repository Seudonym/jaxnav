from jaxnav.optim import cost
from jaxnav.constants import HORIZON, NUM_SAMPLES, LR
from jaxnav.plot import plot_unicycle_path, plot_fan, plot_optimization_result
from jaxnav.model import rollout
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def main():
    state0 = jnp.array([0.0, 0.0, 0.0])
    v, w = 0.5, 0.5
    # lax.scan test
    # state0 = jnp.array([0.0, 0.0, 0.0])
    # v, w = 0.5, 0.5
    #
    # circle_controls = jnp.tile(jnp.array([v, w]), (100, 1))
    #
    # rollout_jit = jax.jit(rollout)
    # circle = jax.block_until_ready(rollout_jit(state0, circle_controls))
    #
    # x, y, theta = circle[:, 0], circle[:, 1], circle[:, 2]
    #
    # plot_unicycle_path(x, y, theta, state0, v, w)

    # vmap test
    # key = jax.random.PRNGKey(0)
    # nominal = jnp.tile(jnp.array([v, 0.0]), (HORIZON, 1))
    # noise_scale = jnp.array([0.1, 0.8])
    # noise = jax.random.normal(key, shape=(NUM_SAMPLES, HORIZON, 2)) * noise_scale
    #
    # batch_controls = nominal + noise
    # batch_rollout =  jax.vmap(rollout, in_axes=(None, 0))
    #
    # fan = batch_rollout(state0, batch_controls)
    # print(f"fan shape: {fan.shape}")
    # plot_fan(fan, state0, nominal)
    
    # optimization
    controls = jnp.tile(jnp.array([v, 0.0]), (HORIZON, 1))
    initial = controls
    goal = jnp.array([5.0, 5.0])
    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=1))

    for i in range(50):
        c, g = cost_and_grad(state0, controls, goal)
        controls = controls - LR * g
        if i % 10 == 0:
            print(f"iter {i:2d}  cost {c:.4f}")
        
    plot_optimization_result(state0, initial, controls, goal)
