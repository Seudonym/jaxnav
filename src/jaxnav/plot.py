from jaxnav.model import rollout
from jaxnav.constants import NUM_SAMPLES
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_unicycle_path(x, y, theta, state0, v, w):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, lw=2, label="path")
    ax.quiver(x[::10], y[::10], jnp.cos(theta[::10]), jnp.sin(theta[::10]),
              angles="xy", scale=15, width=0.006, color="tab:red", label="heading")
    ax.scatter(*state0[:2], color="black", zorder=3, label="start")
    ax.set_aspect("equal")             
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Unicycle with v={v} m/s, ω={w} rad/s")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()

def plot_fan(fan, state0, nominal):
    fig, ax = plt.subplots(figsize=(7, 6))
    for traj in jnp.asarray(fan):
        ax.plot(traj[:, 0], traj[:, 1], lw=0.6, alpha=0.3, color="tab:blue")
    nominal_path = jnp.asarray(rollout(state0, nominal))
    ax.plot(nominal_path[:, 0], nominal_path[:, 1], lw=2, color="black", label="nominal")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{NUM_SAMPLES} noisy rollouts from one vmap call")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()
