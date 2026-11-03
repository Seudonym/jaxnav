from jaxnav.model import rollout
import jax
import jax.numpy as jnp

def cost(state0, controls, goal):
    states = rollout(state0, controls)
    final_xy = states[-1, :2]
    goal_error = jnp.sum((final_xy - goal) ** 2)
    control_effort = 1e-3 * jnp.sum(controls ** 2)
    return goal_error + control_effort

