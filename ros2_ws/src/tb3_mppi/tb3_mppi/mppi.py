import jax
import jax.numpy as jnp

DT = 0.1          
H = 30            
K = 512           
LAMBDA = 1.0      
V_MAX, W_MAX = 0.22, 2.0          
NOISE_STD = jnp.array([0.05, 0.6])
ROBOT_R = 0.12    
MARGIN = 0.10    


def step(s, u):
    x, y, th = s
    v, w = u
    return jnp.array([x + v * jnp.cos(th) * DT,
                      y + v * jnp.sin(th) * DT,
                      th + w * DT])

def rollout(s0, U):
    def body(s, u):
        s1 = step(s, u)
        return s1, s1
    _, traj = jax.lax.scan(body, s0, U)
    return traj # (H, 3)

def cost(traj, U, goal, obstacles):
    # obstacles: (M, 2) points. distance from every trajectory point to every obstacle point
    d = jnp.linalg.norm(traj[:, None, :2] - obstacles[None, :, :], axis=-1) # (H, M)
    min_d = d.min(axis=1) # (H,)

    collision = 1e4 * jnp.any(min_d < ROBOT_R)                       
    proximity = 1e3 * jnp.sum(jnp.maximum(0.0, ROBOT_R + MARGIN - min_d) ** 2)  
    to_goal = jnp.linalg.norm(traj[:, :2] - goal, axis=1)
    goal_cost = jnp.sum(to_goal) + 10.0 * to_goal[-1]                
    smooth = 0.1 * jnp.sum(jnp.diff(U, axis=0) ** 2)                

    return collision + proximity + goal_cost + smooth

@jax.jit
def mppi_step(key, state, U_nom, goal, obstacles):
    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, (K, H, 2)) * NOISE_STD
    U = U_nom + noise
    U = jnp.clip(U, jnp.array([-V_MAX, -W_MAX]), jnp.array([V_MAX, W_MAX]))   

    trajs = jax.vmap(rollout, in_axes=(None, 0))(state, U) # (K, H, 3)
    costs = jax.vmap(cost, in_axes=(0, 0, None, None))(trajs, U, goal, obstacles)

    w = jax.nn.softmax(-costs / LAMBDA)               
    U_new = jnp.sum(w[:, None, None] * U, axis=0)     

    U_next = jnp.roll(U_new, -1, axis=0).at[-1].set(U_new[-1])   
    return key, U_new[0], U_next, trajs, costs
