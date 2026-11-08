# jaxnav

Differentiable and sampling-based motion planning for a differential-drive robot, written in [JAX](https://github.com/jax-ml/jax). The repo goes from a unicycle model, to gradient-based trajectory optimization, to a GPU-accelerated MPPI controller that avoids obstacles on a TurtleBot3 in ROS 2. Its a test repo I used while learning JAX, so `src` is messy.

## Demo

TurtleBot3 (burger) navigating to a goal while avoiding obstacles using the MPPI controller in `ros2_ws/src/tb3_mppi`.
MPPI, localization is running on the robot. The controller is running at 10Hz.

## What's inside

| Path | What it is |
| --- | --- |
| `src/jaxnav/` | Dev library: unicycle dynamics, batched rollouts, gradient-descent trajectory optimizer, plotting used for development |
| `notebooks/` | Jupyter notebooks for experimenting |
| `ros2_ws/src/tb3_mppi/` | ROS 2 (Humble) package running a JAX MPPI controller on a TurtleBot3 deployed and tested |
| `docker/` | ROS 2 Humble + TurtleBot3 + Nav2 + JAX (CUDA) image |

### Core library (`src/jaxnav`)

- **`model.py`**: unicycle model `(x, y, θ)` with controls `(v, ω)`. `rollout` integrates a control sequence with `jax.lax.scan`, and `jax.vmap` batches it over many sampled sequences.
- **`optim.py`**: a cost (squared distance to goal at the final state plus a small control-effort term). Because the whole rollout is differentiable, `jax.value_and_grad` gives the gradient with respect to the controls directly.
- **`plot.py`**: plots a single path, a fan of noisy rollouts, and initial vs. optimized trajectories.
- **`constants.py`**: `DT`, `HORIZON`, `NUM_SAMPLES`, `LR`.

`jaxnav.main()` runs 50 steps of gradient descent on the controls toward the goal `(5, 5)` and plots the result.

### MPPI controller (`ros2_ws/src/tb3_mppi`)

Model Predictive Path Integral control, fully jitted in JAX (`mppi.py`):

1. Sample `K = 512` noisy control sequences around the nominal one, over a horizon of `H = 30` steps at `DT = 0.1 s`. Controls are clipped to the burger's limits (`v ≤ 0.22 m/s`, `|ω| ≤ 2 rad/s`).
2. Roll out all samples in parallel with `vmap(rollout)`.
3. Score each trajectory with `vmap(cost)`:
   - a large penalty for collision (within the robot radius of any obstacle point),
   - a quadratic penalty inside a safety margin,
   - distance to goal along the trajectory, with extra weight on the final point,
   - a control smoothness term.
4. Combine the samples with softmax weights `exp(-cost/λ)`, apply the first control, and shift the sequence forward to warm-start the next step.

`mppi_node.py` wraps this in a ROS 2 node (`jax_mppi`):
