import numpy as np
from scipy.linalg import expm, block_diag
from enum import Enum

####################################################################################################
#               Simulation Classes/Structs to make things easier
####################################################################################################
"""Initialize Noise --> Simulation(with noise), Thrusters --> Spacecraft (with thrusters)"""

class ThrusterParams:
    def __init__(self, thrust_N: float, m_dot_kg_s: float):
        self.thrust_N = thrust_N
        self.m_dot_kg_s = m_dot_kg_s
        self.accel_m_s2 = None

class SatParams:
    def __init__(self, mass_kg: float,
                 thrusters: list[ThrusterParams]):
        self.mass_kg = mass_kg
        self.thrusters = thrusters
        
        for t in self.thrusters:
            t.accel_m_s2 = t.thrust_N / self.mass_kg


# sigma_accel controls the strength of process noise Q.
# Increase to make EKF trust sensors more. Decrease to trust dynamics more.
class NoiseParams:
    def __init__(self, sigma_accel: float, sigma_pos: float, sigma_vel: float):
        self.sigma_accel = sigma_accel  # TUNE THIS FOR PROCESS NOISE Q
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

class SimParams:
    def __init__(self, dt: float,
                 n_steps: int,
                 noise: NoiseParams,
                 mean_motion_rad_s = 0.0011):
        self.dt = dt
        self.N = n_steps
        self.t_tot_sec = dt * n_steps
        self.t_tot_min = self.t_tot_sec / 60.0
        self.noise = noise
        self.n = mean_motion_rad_s

        print(f"Created simulation lasting {self.t_tot_sec:.2f} sec ({self.t_tot_min:.2f} min) with {self.N} steps")

class SimType(Enum):
    mass_varying_with_thrust = 0
    mass_constant = 1

####################################################################################################
#               Clohessy–Wiltshire Equations
####################################################################################################

# Build the CW Matrices
def hcw_continuous_AB(n):
    A = np.zeros((6,6)) # State: x = [x, y, z, xdot, ydot, zdot]^T in LVLH
    B = np.zeros((6,3)) # Input: u = [ax, ay, az]^T specific acceleration in LVLH
    
    # kinematics (position derivatives)
    A[0,3] = 1.0 # row 0 column 3
    A[1,4] = 1.0
    A[2,5] = 1.0
    
    # HCW accelerations (dynamics)
    # ẍ - 2 n ẏ - 3 n^2 x = ax
    # ÿ + 2 n ẋ           = ay
    # z̈ +     n^2 z       = az
    # Written in first-order form, these give the rows for [vx, vy, vz]dot
    A[3, 0] = 3 * n**2   # 3 n^2 x term
    A[3, 4] = 2 * n      # + 2 n ydot term
    A[4, 3] = -2 * n     # - 2 n xdot term
    A[5, 2] = -n**2      # - n^2 z term
    # I’m modeling u = [ax, ay, az] as a direct specific acceleration in LVLH.
    # That means it directly appears in the acceleration states (vx_dot, vy_dot, vz_dot)
    B[3:, :] = np.eye(3)
    return A, B

# Continuous-to-discrete conversion using the Van Loan method.
# This gives me (Ad, Bd) for:
# x_{k+1} = Ad x_k + Bd u_k
# from the continuous system:
# xdot = A x + B u
def c2d_van_loan(A, B, dt):
    # Discretize using Van Loan
    # Build the augmented matrix:
    #   M = [ A  B ]
    #       [ 0  0 ]
    # and scale by dt. The matrix exponential of this block gives Ad and Bd
    M = np.block([
        [A, B],
        [np.zeros((3,6)), np.zeros((3,3))]
    ]) * dt
    E = expm(M)
    # Extract Ad and Bd from the top blocks:
    # E = [ Ad  Bd ]
    #     [ 0    I ]   (for this construction)
    Ad = E[:6,:6]
    Bd = E[:6,6:]
    return Ad, Bd



#    Idea:
#    If the true dynamics have small random accelerations w ~ N(0, sigma_a^2),
#    then position and velocity both get affected after discretization.
#    The mapping from accel noise → (pos, vel) noise is the standard
#    double-integrator covariance:
#        [ dt^3/3   dt^2/2 ]
#        [ dt^2/2   dt     ]  * sigma_a^2

def Qd_from_accel_white(dt, sigma_a):

    # 2x2 covariance for (pos, vel) driven by white accel noise in ONE axis
    Q1 = np.array([
        [dt**3/3, dt**2/2],
        [dt**2/2, dt     ]
    ]) * (sigma_a**2)

    # Build block-diagonal structure for x, y, z axes.
    # At this point the order is [x, xd, y, yd, z, zd]
    Q_block = block_diag(Q1, Q1, Q1)

    # Reorder rows/cols to match OUR state order: [x, y, z, xd, yd, zd]
    # This permutation swaps (pos, vel) pairs into the correct structure.
    perm = np.array([0, 2, 4,   1, 3, 5])
    Q6 = Q_block[np.ix_(perm, perm)]

    return Q6

def simulate_with_thrust(Ad, Bd, Qd, H, R, x0, thrust, m0, m_dot, dt, N, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    # Store full true trajectory (N+1 states because we include x0)
    X_true = np.zeros((N+1, 6))

    # Store all measurements y_k (size depends on H: 3 for pos-only, 6 for full-state)
    Y_meas = np.zeros((N, H.shape[0]))
    Y_true = np.zeros((N, H.shape[0]))

    # Stores all simulated inputs (acceleration, for example)
    U_meas = np.zeros((N, 3)) # stores applied acceleration [ax, ay, az]

    # Errors

    # Set initial true state
    X_true[0] = x0
    m = m0

    # Main simulation loop: propagate truth and generate measurements
    for k in range(N):

        # Physical quantities
        # new
        if k < 80:
            thrust_cmd = np.array([1.0, 0.0, 0.0]) * -thrust  # or whatever the control wants
            accel_cmd = thrust_cmd / m  # ideal acceleration (m/s^2)

            accel_mag = np.linalg.norm(accel_cmd)
            noise_std = 0.02 * accel_mag
            accel_noise = rng.normal(0.0, noise_std, size=3) if noise_std > 0 else np.zeros(3)

            u = accel_cmd + accel_noise


        else:
            u = np.zeros(3)

        # DEBUG: print first few timesteps to see direction of motion
        if k < 10:
            print(f"[HCW DEBUG] k={k:3d}, x={X_true[k,0]:8.3f} m, u_x={u[0]: .4e} m/s^2")


        # Process noise w_k ~ N(0, Qd)
        # This represents unmodeled forces (J2, drag, thrust errors, etc.)
        w = rng.multivariate_normal(np.zeros(6), Qd)

        

        # True dynamics:
        # x_{k+1} = Ad * x_k + Bd * u_k + w_k
        x_next = Ad @ X_true[k] + Bd @ u + w
        

        # Measurement noise v_k ~ N(0, R)
        v = rng.multivariate_normal(np.zeros(H.shape[0]), R)

        # Measurement model:
        # y_k = H * x_{k+1} + v_k
        # (note: measurement of the NEW state)
        y = H @ x_next

        # Change physical quantities
        m -= m_dot*dt

        # Save results
        X_true[k+1] = x_next
        Y_true[k]   = y
        Y_meas[k]   = y + v
        U_meas[k] = u

    return X_true, Y_true, Y_meas, U_meas

def simulate(Ad, Bd, Qd, H, R, x0, U, N, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    # Store full true trajectory (N+1 states because we include x0)
    X_true = np.zeros((N+1, 6))

    # Store all measurements y_k (size depends on H: 3 for pos-only, 6 for full-state)
    Y_meas = np.zeros((N, H.shape[0]))
    Y_true = np.zeros((N, H.shape[0]))

    # Stores all simulated inputs (acceleration, for example)
    U_meas = np.zeros((N, 3)) # for ax, ay, az

    # Set initial true state
    X_true[0] = x0

    # Main simulation loop: propagate truth and generate measurements
    for k in range(N):

        # Process noise w_k ~ N(0, Qd)
        # This represents unmodeled forces (J2, drag, thrust errors, etc.)
        w = rng.multivariate_normal(np.zeros(6), Qd)

        # True dynamics:
        # x_{k+1} = Ad * x_k + Bd * u_k + w_k
        x_next = Ad @ X_true[k] + Bd @ U[k] + w

        # Measurement noise v_k ~ N(0, R)
        v = rng.multivariate_normal(np.zeros(H.shape[0]), R)

        # Measurement model:
        # y_k = H * x_{k+1} + v_k
        # (note: measurement of the NEW state)
        y = H @ x_next

        # Save results
        X_true[k+1] = x_next
        Y_true[k]   = y
        Y_meas[k]   = y + v
        U_meas[k] = U[k]

    return X_true, Y_true, Y_meas, U_meas


####################################################################################################
#               Actual HCW Simulation
####################################################################################################


# NEW CHANGE TO THE CLASS!!!
# ------------------------------------------------------------
# Truth–Model Mismatch Capability
# ------------------------------------------------------------
# This class (below) now supports *different* dynamics for:
#   (1) the truth simulation  – used to propagate the real system
#   (2) the model dynamics    – used by the EKF inside the POMDP
#
# If `model_n` or `model_sigma_accel` are not given, they
# default to the truth values, so there would be no mismatch.
#
# By allowing the model to use slightly incorrect parameters,
# we simulate a realistic scenario where the filter and controller
# do NOT have perfect knowledge of the true orbital dynamics.
# This makes the rendezvous problem more realistic and strengthens
# the research contribution.
# ------------------------------------------------------------


class HCWDynamics_3DOF:
    def __init__(self, simulation: SimParams, satellite: SatParams,
                 state0: np.ndarray, model_n: float = None, model_sigma_accel: float = None):
        
        self.sim = simulation
        self.sat = satellite
        self.state0 = state0
        
        n_truth = self.sim.n  # change to have “truth” HCW parameters and “model” HCW parameters
        dt = self.sim.dt
        sigma_truth = self.sim.noise.sigma_accel

        # ---- TRUTH DYNAMICS (used in simulate) ----
        A_truth, B_truth = hcw_continuous_AB(n_truth)
        Ad_truth, Bd_truth = c2d_van_loan(A_truth, B_truth, dt)
        Qd_truth = Qd_from_accel_white(dt, sigma_truth)

        # store as the “default” ones for simulation
        self.A      = A_truth
        self.B      = B_truth
        self.Ad     = Ad_truth
        self.Bd     = Bd_truth
        self.Qd     = Qd_truth
        self.t_array = np.arange(self.sim.N + 1) * dt

        # ---- MODEL DYNAMICS (for EKF, possibly mismatched) ----
        if model_n is None:
            model_n = n_truth      # default: no mismatch in n
        if model_sigma_accel is None:
            model_sigma_accel = sigma_truth  # default: no mismatch in Q

        A_model, B_model = hcw_continuous_AB(model_n)
        Ad_model, Bd_model = c2d_van_loan(A_model, B_model, dt)
        Qd_model = Qd_from_accel_white(dt, model_sigma_accel)

        # store separately so EKF can use them later
        self.A_model  = A_model
        self.B_model  = B_model
        self.Ad_model = Ad_model
        self.Bd_model = Bd_model
        self.Qd_model = Qd_model



    def simulate(self, H: np.ndarray, R: np.ndarray, U: np.ndarray = None, rng_seed = 42, simtype: SimType = SimType.mass_varying_with_thrust):

        # Will be many more cases for simulation, but there are only a couple right now
        if simtype == SimType.mass_varying_with_thrust:
            X_true, Y_true, Y_meas, U_meas = simulate_with_thrust(
                self.Ad, self.Bd, self.Qd, H, R, self.state0,
                self.sat.thrusters[0].thrust_N, self.sat.mass_kg,
                self.sat.thrusters[0].m_dot_kg_s, self.sim.dt,
                self.sim.N, rng_seed
            )
        elif simtype == SimType.mass_constant:
            if U is None:
                raise ValueError("For non-thrust simulation, must input U")
            X_true, Y_true, Y_meas, U_meas = simulate(
                self.Ad, self.Bd, self.Qd, H, R, self.state0,
                U, self.sim.N, rng_seed
            )
        else:
            raise NotImplementedError("Sim type is not implemented")

        self.X_true, self.Y_true, self.Y_meas, self.U_meas = X_true, Y_true, Y_meas, U_meas

        # Project true state into measurement space so sizes match
        self.state_err = Y_meas - (H @ X_true[1:].T).T  # shape (N, meas_dim)

        print("Simulated!")

        return X_true, Y_true, Y_meas, self.state_err, U_meas