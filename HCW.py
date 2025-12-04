import numpy as np
from scipy.linalg import expm, block_diag
from enum import Enum
from collections.abc import Callable
from scipy.linalg import norm

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

class NoiseParams:
    def __init__(self, sigma_accel: float, sigma_pos: float, sigma_vel: float):
        self.sigma_accel = sigma_accel
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

class SimParams:
    def __init__(self, dt: float,
                 n_steps: float,
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

class HCW_Params:
    Ad: np.ndarray
    Bd: np.ndarray
    Qd: np.ndarray
    H: np.ndarray
    R: np.ndarray

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
    U_meas = np.zeros((N, 3)) # for ax, ay, az

    # Errors

    # Set initial true state
    X_true[0] = x0
    m = m0

    # Main simulation loop: propagate truth and generate measurements
    for k in range(N):

        # Physical quantities
        if k < 80:
            u = np.array([1,0,0]) * -thrust/m
        else:
            u = np.zeros(3)


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

def hill_derivative(state: np.ndarray, input_N: np.ndarray, mass_kg: float, mean_motion: float):
    """Simulates one step according to hill's differential equation.
    I want to see if this gives the same accuracy as doing the discrete Ad and Bd

    Args:
        state (np.ndarray): [x, y, z, vx, vy, vz]
        input (np.ndarray): [Fx, Fy, Fz]

    Returns:
        np.ndarray: [vx, vy, vz, ax, ay, az]
    """
    x, _, z, vx, vy, vz = state
    Fx, Fy, Fz = input_N
    n = mean_motion
    m = mass_kg

    ax = (3*n*n*x + 2*n*vy) + Fx/m
    ay = (-2*n*vx) + Fy/m
    az = (-n*n*z) + Fz/m

    return np.array([vx, vy, vz, ax, ay, az])

def rk4(dt: float, x: np.ndarray, x_dot_func: Callable[[float, np.ndarray], np.ndarray],
        input_N: float, mass_kg: float, mean_motion: float):
    """Baby version of RK4 (no time, much more inputs)"""
    k1 = x_dot_func(x, input_N, mass_kg, mean_motion)
    k2 = x_dot_func(x + 0.5*dt*k1, input_N, mass_kg, mean_motion)
    k3 = x_dot_func(x + 0.5*dt*k2, input_N, mass_kg, mean_motion)
    k4 = x_dot_func(x + dt*k3, input_N, mass_kg, mean_motion)

    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_step_discrete(state: np.ndarray, input_N: np.ndarray,
                           hcw_params: HCW_Params,
                           rng_seed: int = 42
                           ):
    """Essentially turns a (s, a) -> (s', o). Reward is calculated after. Use HCW_Params"""

    Ad, Bd, Qd, R, H = hcw_params.Ad, hcw_params.Bd, hcw_params.Qd, hcw_params.R, hcw_params.H
    rng = np.random.default_rng(rng_seed)

    # Process noise w_k ~ N(0, Qd)
    # This represents unmodeled forces (J2, drag, thrust errors, etc.)
    w = rng.multivariate_normal(len(Ad), Qd)

    # True dynamics:
    # x_{k+1} = Ad * x_k + Bd * u_k + w_k
    next_state = Ad @ state + Bd @ input_N + w

    # Measurement noise v_k ~ N(0, R)
    v = rng.multivariate_normal(len(input_N), R)

    # Measurement model:
    # y_k = H * x_{k+1} + v_k
    # (note: measurement of the NEW state)
    measurement = H @ next_state

    # Returns:
    # - true state
    # - measurement
    # - true measurement (for comparison)
    return (next_state,
            measurement,
            measurement + v)

def reward_1(next_state: np.ndarray, input_N: np.ndarray, m0: float, mass_kg: float):
    """(s',a) --> (r)   Should the reward also be a function of action too?"""
    s, _ = next_state, input_N

    r = norm(s[:3])
    v = norm(s[3:6])
    R_dist = -r

    # If closer than 100 meters, give reward for slow behavior
    R_vel = 0 if r > 100 else -np.sqrt(v)

    # Fuel reward
    R_mass = 100 * mass_kg/m0 - 50
    

    return R_dist + R_vel + R_mass

    

####################################################################################################
#               Actual HCW Simulation
####################################################################################################

class HCWDynamics_3DOF:
    def __init__(self, simulation: SimParams, satellite: SatParams,
                 state0: np.ndarray):
        
        self.sim = simulation
        self.sat = satellite
        self.state0 = state0
        
        n, dt = self.sim.n, self.sim.dt
        sigma_accel = self.sim.noise.sigma_accel

        A, B = hcw_continuous_AB(n)
        Ad, Bd = c2d_van_loan(A, B, dt)
        Qd = Qd_from_accel_white(dt, sigma_accel)

        self.A, self.B, self.Ad, self.Bd, self.Qd = A, B, Ad, Bd, Qd
        self.t_array = np.arange(self.sim.N + 1) * dt

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