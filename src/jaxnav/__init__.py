from jaxnav.optim import cost
from jaxnav.constants import HORIZON, NUM_SAMPLES, LR
from jaxnav.plot import plot_unicycle_path, plot_fan
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
    cost_grad = jax.grad(cost, argnums=1)
    controls = jnp.tile(jnp.array([v, 0.0]), (HORIZON, 1))
    initial = controls
    goal = jnp.array([5.0, 5.0])

    grad = cost_grad(state0, controls, goal)
    print(f"grad shape: {grad.shape}")

    for i in range(50):
        controls = controls - LR * cost_grad(state0, controls, goal)
        if i % 10 == 0:
            print(f"iter {i:2d}  cost {cost(state0, controls, goal):.4f}")
        

    before = jnp.asarray(rollout(state0, initial))
    after = jnp.asarray(rollout(state0, controls))
    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(before[:, 0], before[:, 1], "--", color="gray", label="initial guess")
    ax.plot(after[:, 0], after[:, 1], lw=2, color="tab:green", label="optimized")
    ax.scatter(*goal, marker="*", s=250, color="tab:red", zorder=3, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Controls optimized with jax.grad")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()

