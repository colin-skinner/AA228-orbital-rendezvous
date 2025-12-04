# Closed-loop rendezvous simulation:
# HCW dynamics + EKF state estimation + POMDP/MCTS decision-making.

# This is the systems code that ties together:
# - HCW.py   (dynamics + process noise)
# - ekf.py   (state estimation)
# - pomdp.py (planning over beliefs)

import numpy as np

from HCW import (
    SimParams,
    SatParams,
    NoiseParams,
    ThrusterParams,
    HCWDynamics_3DOF,
)
from ekf import LinearEkf, State, run_linear_ekf
from pomdp import POMDP, MCTS


# -----------------------------
# Action space + mapping
# -----------------------------
# I’m modeling a small discrete action space:
# 0 = coast (no thrust)
# 1 = fine thrust along -x (small accel toward target)
# 2 = coarse thrust along -x (larger accel toward target)
#
# This function maps an integer action into a specific acceleration
# vector in LVLH, using the satellite's thruster parameters.
def action_to_accel(action: int, sat: SatParams) -> np.ndarray:
    if action == 0:
        return np.zeros(3)

    # We assume thrusters[0] = fine, thrusters[1] = coarse
    if action == 1:
        ax = -sat.thrusters[0].accel_m_s2
        return np.array([ax, 0.0, 0.0])

    if action == 2:
        ax = -sat.thrusters[1].accel_m_s2
        return np.array([ax, 0.0, 0.0])

    raise ValueError(f"Unknown action {action}")


# -----------------------------
# Reward function
# -----------------------------
# Simple quadratic stage cost:
#   - penalize distance to target
#   - penalize relative velocity
#   - penalize thrust usage
# plus a docking bonus if inside a small box around the target.
def compute_reward(
    x_next: np.ndarray,
    u_cmd: np.ndarray,
    w_pos: float = 1.0,
    w_vel: float = 0.1,
    w_u: float = 0.01,
    dock_bonus: float = 100.0,
    dock_tol_pos: float = 0.5,
    dock_tol_vel: float = 0.05,
) -> float:
    # State: [x, y, z, xdot, ydot, zdot]
    pos = x_next[:3]
    vel = x_next[3:]

    pos_cost = w_pos * np.dot(pos, pos) # Position penalty: penalizes from being far from the target, sp we get closer
    vel_cost = w_vel * np.dot(vel, vel) # velocity penalty: Penalizes having high relative speed
    # Encourages the spacecraft to slow down as it gets closer to the target
    u_cost = w_u * np.dot(u_cmd, u_cmd) # Control (thrust) penalty: encourages fuel efficiency, so penalizes thrust amount

    r = -(pos_cost + vel_cost + u_cost)

    # Docking bonus if we're very close and slow
    if np.linalg.norm(pos) < dock_tol_pos and np.linalg.norm(vel) < dock_tol_vel:
        r += dock_bonus
    # Impact penalty if we're close and too fast    
    if np.linalg.norm(pos) < dock_tol_pos and np.linalg.norm(vel) > v_impact_thresh:
        r -= big_crash_penalty

    return r

    # Reward = negative cost + docking bonus
    # Far + fast + high thrust = bad
    # Close + slow + low thrust = good
    # Docked = very good


# -----------------------------
# Main closed-loop simulation
# -----------------------------
def run_closed_loop_episode(
    dt: float = 1.0,
    N: int = 300,
    mean_motion: float = 0.0011,
    sigma_accel_truth: float = 1e-5,
    sigma_accel_model: float = 2e-5,
    sigma_meas_pos: float = 1.0,
    seed: int = 42,
):
    """
    Run one closed-loop rendezvous episode.

    Returns a dict with:
      - X_true:   (N+1, 6) true states
      - X_hat:    (N+1, 6) EKF estimated states
      - actions:  (N,)    chosen actions at each step
      - rewards:  (N,)    reward per step
      - pos_error_norm: (N+1,) position error norm ||pos||
    """
    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1) Set up dynamics + satellite
    # -----------------------------
    noise = NoiseParams(
        sigma_accel=sigma_accel_truth,
        sigma_pos=0.0,
        sigma_vel=0.0,
    )

    sim = SimParams(
        dt=dt,
        n_steps=int(N),
        noise=noise,
        mean_motion_rad_s=mean_motion,
    )

    # Two thrusters: fine and coarse
    # (These values are arbitrary; we can tune them later.)
    thrusters = [
        ThrusterParams(thrust_N=0.05, m_dot_kg_s=1e-4),   # fine
        ThrusterParams(thrust_N=0.2, m_dot_kg_s=4e-4),    # coarse
    ]
    mass_kg = 500.0
    sat = SatParams(mass_kg=mass_kg, thrusters=thrusters)

    # Initial relative state: offset along +x, at rest
    x0_true = np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Build dynamics object with truth/model mismatch in Q and n
    hcw = HCWDynamics_3DOF(
        simulation=sim,
        satellite=sat,
        state0=x0_true,
        model_n=mean_motion,                # could tweak separately
        model_sigma_accel=sigma_accel_model # model believes a different accel noise
    )

    # Truth dynamics for environment
    Ad_truth = hcw.Ad
    Bd_truth = hcw.Bd
    Qd_truth = hcw.Qd

    # Model dynamics for EKF
    Ad_model = hcw.Ad_model
    Bd_model = hcw.Bd_model
    Qd_model = hcw.Qd_model

    # -----------------------------
    # 2) Measurement model
    # -----------------------------
    # For now, measure position only: y = [x, y, z]^T
    H = np.zeros((3, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0

    R = (sigma_meas_pos**2) * np.eye(3)

    # -----------------------------
    # 3) EKF setup
    # -----------------------------
    # Initial EKF state: slightly wrong initial guess (for realism)
    x0_hat = x0_true + np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    P0 = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])

    ekf_model = LinearEkf(
        F=Ad_model,
        B=Bd_model,
        H=H,
        Q=Qd_model,
        R=R,
    )
    belief = State(x=x0_hat.copy(), P=P0.copy())

    # -----------------------------
    # 4) POMDP + MCTS setup
    # -----------------------------
    # Discrete actions: 0, 1, 2
    actions = [0, 1, 2]

    # We'll define TRO as a closure over our environment parameters.
    def tro_func(s: np.ndarray, a: int):
        """
        One-step environment model for MCTS:
        - s: current true state (6,)
        - a: discrete action index
        Returns: (s_next, r, o)
        """
        # Commanded acceleration based on action (no randomness here)
        u_cmd = action_to_accel(a, sat)

        # Process noise on truth
        w = rng.multivariate_normal(mean=np.zeros(6), cov=Qd_truth)

        # Propagate HCW dynamics
        s_next = Ad_truth @ s + Bd_truth @ u_cmd + w

        # Measurement noise
        v = rng.multivariate_normal(mean=np.zeros(H.shape[0]), cov=R)
        o = H @ s_next + v

        # Reward based on next state + command
        r = compute_reward(s_next, u_cmd)

        return s_next, r, o

    # POMDP container: many fields are not used by MCTS in this setup,
    # but we fill them for completeness
    pomdp_problem = POMDP(
        gamma=0.99,
        states=None,           # continuous; not enumerated
        actions=actions,
        observations=None,     # implicit via H and R
        transition=None,       # we use TRO directly
        reward=None,           # computed inside TRO
        observation=None,      # not needed explicitly
        tro=tro_func,
    )

    planner = MCTS(
        problem=pomdp_problem,
        depth=5,
        num_sims=50,
        c=1.0,
        rollout=None,
        rollout_depth=5,
    )

    # -----------------------------
    # 5) Allocate logs
    # -----------------------------
    X_true = np.zeros((N + 1, 6))
    X_hat = np.zeros((N + 1, 6))
    actions_log = np.zeros(N, dtype=int)
    rewards_log = np.zeros(N)
    pos_error_norm = np.zeros(N + 1)

    X_true[0] = x0_true
    X_hat[0] = x0_hat
    pos_error_norm[0] = np.linalg.norm(x0_true[:3])

    # History for MCTS (sequence of (action, observation) pairs)
    h = ()

    # -----------------------------
    # 6) Closed-loop simulation
    # -----------------------------
    s_true = x0_true.copy()

    for k in range(N):
        # Belief passed to planner is the EKF Gaussian state
        belief_for_planner = State(x=belief.x.copy(), P=belief.P.copy())

        # Choose action with MCTS
        a_k = planner(belief_for_planner, h)
        actions_log[k] = a_k

        # Commanded accel (same mapping as in TRO)
        u_cmd = action_to_accel(a_k, sat)

        # True environment step using the same TRO model
        s_next, r_k, o_k = tro_func(s_true, a_k)

        # EKF update with commanded control and the noisy measurement
        belief = run_linear_ekf(
            state=belief,
            ekf=ekf_model,
            u=u_cmd,
            z=o_k,
        )

        # Update history for MCTS (it only cares about (a, o))
        h = h + ((a_k, o_k),)

        # Log states and reward
        s_true = s_next
        X_true[k + 1] = s_true
        X_hat[k + 1] = belief.x
        rewards_log[k] = r_k
        pos_error_norm[k + 1] = np.linalg.norm(s_true[:3])

    results = {
        "X_true": X_true,
        "X_hat": X_hat,
        "actions": actions_log,
        "rewards": rewards_log,
        "pos_error_norm": pos_error_norm,
    }

    return results


# -----------------------------
# test
# -----------------------------
if __name__ == "__main__":
    # Run one episode with default parameters
    res = run_closed_loop_episode()

    X_true = res["X_true"]
    X_hat = res["X_hat"]
    actions = res["actions"]
    rewards = res["rewards"]
    pos_err = res["pos_error_norm"]

    print("Simulation finished.")
    print(f"True state shape:      {X_true.shape}")
    print(f"Estimated state shape: {X_hat.shape}")
    print(f"Number of steps:       {len(actions)}")

    print(f"First 10 actions:      {actions[:10]}")
    print(f"Initial position error: {pos_err[0]:.3f} m")
    print(f"Final position error:   {pos_err[-1]:.3f} m")
    print(f"Total reward:           {np.sum(rewards):.3f}")