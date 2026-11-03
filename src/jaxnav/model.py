import jaxnav.constants
import jax
import jax.numpy as jnp

def step(state, control):
    x, y, theta = state
    v, w = control
    return jnp.array([
        x + v * jnp.cos(theta) * jaxnav.constants.DT,
        y + v * jnp.sin(theta) * jaxnav.constants.DT,
        theta + w * jaxnav.constants.DT
        ])

def rollout(state0, controls):
    def step_fn(state, control):
        next_state = step(state, control)
        return next_state, next_state
    _, states = jax.lax.scan(step_fn, state0, controls)
    return states

