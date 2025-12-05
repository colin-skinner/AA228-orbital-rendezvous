# Closed-loop rendezvous simulation:
# HCW dynamics + EKF state estimation + POMDP/MCTS decision-making
# Author: Sabrina Nicacio

# This is the systems code that ties together:
# - HCW.py   (dynamics + process noise) - Sabrina
# - ekf.py   (state estimation) - Colin
# - pomdp.py (planning over beliefs) - Enrico

import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys # For args
import cProfile

from HCW import (
    SimParams,
    SatParams,
    NoiseParams,
    ThrusterParams,
    HCWDynamics_3DOF,
    hcw_continuous_AB,
    c2d_van_loan,
    Qd_from_accel_white,
)
from ekf import LinearEkf, State, run_linear_ekf
from pomdp import POMDP, MCTS

# -------------------------------------------------------
# Simulation Variables
# -------------------------------------------------------
# Still need to work on tuning the simulation for better results
# All variables are labeled and described so it is easier to understand and change
CONFIG = {
    # --- simulation setup ---
    "dt": 1.0,                     # [s] simulation time step
    "N": 600,                      # number of time steps (horizon = N * dt secs) Can make it longer for better results
    "initial_state": np.zeros(6),  # 6-state vector [x,y,z,xdot,ydot,zdot]

    # --- process / measurement noise scales (for Q and R) ---
    "sigma_pos": 10.0,             # [m] (not used directly here)
    "sigma_vel": 0.1,              # [m/s]
    "sigma_accel": 0.01,           # [m/s^2]
    "R_meas": np.diag([10.0, 10.0, 10.0]),  # legacy; EKF uses sigma_meas_pos below
    "mean_motion": 0.0011,         # [rad/s] HCW mean motion n

    "sigma_accel_truth": 1e-5,     # [m/s^2] accel noise std for TRUE process noise Q
    "sigma_accel_model": 2e-5,     # [m/s^2] accel noise std for EKF model Q
    "sigma_meas_pos": 1.0,         # [m] per-axis position noise std for EKF R

    # --- initial state and belief (EKF) ---
    "x0_true": np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # start 50 m along x
    "x0_hat_offset": np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
    "P0_diag": np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),

    # --- reward shaping for the POMDP / MCTS ---
    # See reward function later in the code
    "reward": {
        "w_pos": 10.0,          # weight on ||position||^2
        "w_vel": 0.05,           # STRONGER weight on ||velocity||^2 to avoid huge speeds
        "w_u": 1e-6,            # weight on ||u||^2
        "dock_bonus": 3000.0,  # big bonus when inside docking tolerances
        "dock_tol_pos": 2.0,    # [m]
        "dock_tol_vel": 0.3,   # [m/s]
        "v_impact_thresh": 1.0, # [m/s] inside tol_pos but faster than this → crash
        "big_crash_penalty": 5000.0,
        "alpha": 20.0,           # weight on progress term (prev_dist - new_dist)
    },
}


# -------------- manual HCW dynamics check -------------------
def manual_hcw_test():
    """
    Simple check: if we apply constant thrust toward the target (the origin),
    does the spacecraft actually get closer?
    This does NOT use POMDP or EKF — just the dynamics.
    """

    dt = CONFIG["dt"]
    N = CONFIG["N"]
    n = CONFIG["mean_motion"]

    A, B = hcw_continuous_AB(n)
    Ad, Bd = c2d_van_loan(A, B, dt)

    x = CONFIG["x0_true"].copy()

    # test constant accel toward origin
    coarse_acc = -0.05  # m/s^2 along -x as a sanity check
    u_cmd = np.array([coarse_acc, 0.0, 0.0])

    for _ in range(N):
        x = Ad @ x + Bd @ u_cmd

    dist0 = np.linalg.norm(CONFIG["x0_true"][:3])
    dist1 = np.linalg.norm(x[:3])

    print("\n====== MANUAL HCW TEST ======")
    print(f"Initial distance: {dist0:.3f} m")
    print(f"Final distance:   {dist1:.3f} m")
    print(f"Change:           {dist1 - dist0:.3f} m")
    print("=============================\n")


# -----------------------------
# Action space + mapping
# -----------------------------
# 0 = coast (no thrust)
# 1 = fine thrust along -x
# 2 = coarse thrust along -x
def action_to_accel(action: int, sat: SatParams) -> np.ndarray:
    if action == 0:
        return np.zeros(3)

    # thrusters[0] = fine, thrusters[1] = coarse
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
def compute_reward(x_prev: np.ndarray, x_next: np.ndarray, u_cmd: np.ndarray) -> float:
    """
    Reward shaping for docking.

    - **Only** care about PROGRESS toward the target and small control effort.
    - Do NOT penalize absolute distance directly (that was killing the planner).
    - Big bonus if we are close AND slow (docked).
    - Big penalty if we are close but too fast (impact).
    """
    r_cfg = CONFIG["reward"]

    w_u               = r_cfg["w_u"]
    dock_bonus        = r_cfg["dock_bonus"]
    dock_tol_pos      = r_cfg["dock_tol_pos"]
    dock_tol_vel      = r_cfg["dock_tol_vel"]
    v_impact_thresh   = r_cfg["v_impact_thresh"]
    big_crash_penalty = r_cfg["big_crash_penalty"]
    alpha             = r_cfg["alpha"]

    # State: [x, y, z, xdot, ydot, zdot]
    pos_prev = x_prev[:3]
    pos_next = x_next[:3]
    vel_next = x_next[3:]

    # ---- 1. Progress term: positive if we move closer, negative if we move away ----
    prev_dist = np.linalg.norm(pos_prev)
    new_dist  = np.linalg.norm(pos_next)
    progress  = prev_dist - new_dist          # >0 if closer, <0 if farther
    r = alpha * progress                      # main driver of behavior

    # ---- 2. Small control penalty (fuel use) ----
    u_cost = w_u * np.dot(u_cmd, u_cmd)
    r -= u_cost

    # ---- 3. Docking bonus: very close AND slow ----
    if (np.linalg.norm(pos_next) < dock_tol_pos and
        np.linalg.norm(vel_next) < dock_tol_vel):
        r += dock_bonus

    # ---- 4. Impact penalty: close but too fast ----
    if (np.linalg.norm(pos_next) < dock_tol_pos and
        np.linalg.norm(vel_next) > v_impact_thresh):
        r -= big_crash_penalty

    return r


# -----------------------------
# Main closed-loop sim
# -----------------------------
def run_closed_loop_episode(
    dt: float = CONFIG["dt"],
    N: int = CONFIG["N"],
    mean_motion: float = CONFIG["mean_motion"],
    sigma_accel_truth: float = CONFIG["sigma_accel_truth"],
    sigma_accel_model: float = CONFIG["sigma_accel_model"],
    sigma_meas_pos: float = CONFIG["sigma_meas_pos"],
    seed: int = 42,
    debug = True
):
    """
    Run one closed-loop rendezvous episode.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------
    # 1) Set up dynamics + satellite
    # ------------------------------
    noise = NoiseParams(
        sigma_accel=sigma_accel_truth,
        sigma_pos=0.0,
        sigma_vel=0.0,
    )

    sim = SimParams(
        dt=CONFIG["dt"],
        n_steps=N,
        noise=noise,
        mean_motion_rad_s=CONFIG["mean_motion"],
    )

    # Two thrusters: fine and coarse
    thrusters = [
        ThrusterParams(thrust_N=5.0, m_dot_kg_s=1e-3),   # fine
        ThrusterParams(thrust_N=20.0, m_dot_kg_s=4e-3),  # coarse
    ]
    # Larger mass → smaller accelerations → easier to control
    mass_kg = 500.0
    sat = SatParams(mass_kg=mass_kg, thrusters=thrusters)

    # Initial true state
    x0_true = CONFIG["x0_true"].copy()

    # Build dynamics object with truth/model mismatch in Q and n
    hcw = HCWDynamics_3DOF(
        simulation=sim,
        satellite=sat,
        state0=x0_true,
        model_n=mean_motion,
        model_sigma_accel=sigma_accel_model,
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
    H = np.zeros((3, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0

    R = (sigma_meas_pos**2) * np.eye(3)

    # -----------------------------
    # 3) EKF
    # -----------------------------
    x0_hat = x0_true + CONFIG["x0_hat_offset"]
    P0 = np.diag(CONFIG["P0_diag"])

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
    actions = np.array([0, 1, 2])
    r_cfg = CONFIG["reward"]
    dock_tol_pos = r_cfg["dock_tol_pos"]
    dock_tol_vel = r_cfg["dock_tol_vel"]

    def tro_func(s: np.ndarray, a: int):
        """
        One-step environment model for MCTS.
        Implements an absorbing docked state: once docked, state stops moving
        and additional actions have no effect.
        """
        pos = s[:3]
        vel = s[3:]

        # If already docked, stay there (absorbing state)
        if np.linalg.norm(pos) < dock_tol_pos and np.linalg.norm(vel) < dock_tol_vel:
            s_next = s.copy()
            u_zero = np.zeros(3)
            o = H @ s_next
            r = 0.0  # no extra cost/bonus after docking
            return s_next, r, o

        # Otherwise, apply commanded acceleration
        u_cmd = action_to_accel(a, sat)

        # No process / measurement noise for now (deterministic for MCTS)
        w = np.zeros(6)

        # Propagate HCW dynamics
        s_next = Ad_truth @ s + Bd_truth @ u_cmd + w

        # If we JUST docked this step, clamp exactly to zero state
        pos_next = s_next[:3]
        vel_next = s_next[3:]
        if np.linalg.norm(pos_next) < dock_tol_pos and np.linalg.norm(vel_next) < dock_tol_vel:
            s_next[:3] = 0.0
            s_next[3:] = 0.0

        o = H @ s_next

        # Reward based on next state + command
        r = compute_reward(s, s_next, u_cmd)

        return s_next, r, o

    pomdp_problem = POMDP(
        gamma=0.99,
        states=None,
        actions=actions,
        observations=None,
        transition=None,
        reward=None,
        observation=None,
        tro=tro_func,
    )

    planner = MCTS(
        problem=pomdp_problem,
        depth=8,       # slightly deeper planning horizon
        num_sims=80,   # more simulations for better decisions
        c=1.0,
        rollout=None,
        rollout_depth=5,
    )

    # -----------------------------
    # 5) logs
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


    # ----------------------------------
    # Simple heuristic docking override
    # ----------------------------------
    # Needed to add a controller because the results were really bad without it
    def docking_heuristic(s: np.ndarray, a_mcts: int) -> int:
        """
        Very simple controller that overrides MCTS if needed.
        Uses true state s = [x,y,z,xd,yd,zd].

        Goal:
        - If we are far in +x, thrust along -x (action 2).
        - When we are closer, switch to fine thrust (action 1).
        - If we are inside docking box and slow, coast (action 0).
        """
        x = s[0]
        vx = s[3]

        dock_cfg = CONFIG["reward"]
        pos_tol = dock_cfg["dock_tol_pos"]
        vel_tol = dock_cfg["dock_tol_vel"]

        # If already docked region: coast
        if abs(x) < pos_tol and abs(vx) < vel_tol:
            return 0

        # If we are still many meters away, push hard toward origin
        if x > 5.0:
            return 2   # coarse thrust along -x
        if x > pos_tol:
            return 1   # fine thrust along -x

        # If we overshoot to negative x, gently push back
        if x < -5.0:
            return 2   # thrust along +x (we'll interpret sign via action_to_accel if needed)
        if x < -pos_tol:
            return 1

        # Otherwise, use whatever MCTS suggested
        return a_mcts

    # -----------------------------
    # 6) Closed-loop simulation
    # -----------------------------
    s_true = x0_true.copy()

    for k in range(N):
        # Belief passed to planner
        belief_for_planner = State(x=belief.x.copy(), P=belief.P.copy())

        # Choose action with MCTS
        belief_for_planner = State(x=belief.x.copy(), P=belief.P.copy())
        a_k = planner(belief_for_planner, h)
        actions_log[k] = a_k

        # ---- DOCKING STOP CONDITION (PASTE THIS HERE) ----
        pos = s_true[:3]
        vel = s_true[3:]
        dock_cfg = CONFIG["reward"]

        if (np.linalg.norm(pos) < dock_cfg["dock_tol_pos"] and
            np.linalg.norm(vel) < dock_cfg["dock_tol_vel"]):
            # Once docked, force coast action forever
            a_k = 0
            actions_log[k] = 0
    # ---------------------------------------------------

        # Commanded accel
        u_cmd = action_to_accel(a_k, sat)

        # True environment step (same tro model)
        s_next, r_k, o_k = tro_func(s_true, a_k)

        # ---- DEBUG: first few steps
        if debug and k % 10 == 0:
            print(
                f"[DEBUG] k={k}, x = {s_true[0]:8.3f} m, "
                f"u_x = {u_cmd[0]:+8.3e} m/s^2, "
                f"x_next = {s_next[0]:8.3f} m, "
                f"r = {r_k:8.3f}"
            )

        # EKF update
        belief = run_linear_ekf(
            state=belief,
            ekf=ekf_model,
            u=u_cmd,
            z=o_k,
        )

        # Update history for MCTS
        h = h + ((a_k, o_k),)

        # Log
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
        "N": N
    }

    return results


# -------------------------------------------------------------------------------------------------------------------------------------------------
#               MAIN FUNCTION
# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    def main():
        action = "run"
        cache_name = "cache"
        debug = False

        args = sys.argv

        if len(args) > 1:
            action = args[1]

        if len(args) > 2:
            cache_name == str(args[2])
        print(args)
        
        match action:
            case "run":
                res = run_closed_loop_episode(N=100, debug=debug)
                print(f"Ran with {res["N"]} steps")

            case "load":
                print(f"Loading from {cache_name}")
                with open(cache_name,"rb") as f:
                    res = pickle.load(f)
                print(f"Loaded with {res["N"]} steps")

        if "save" in args:
            print(f"Saved in {cache_name}")
            with open(cache_name,"wb") as f:
                pickle.dump(res, f)

        print("\n\n")


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

    cProfile.run('main()')
        # plt.plot(X_true[:, 0])
        # plt.show()

    









    
