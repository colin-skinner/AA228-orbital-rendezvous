import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

@dataclass
class State:
    x: np.ndarray  # Mean vector
    P: np.ndarray  # Covariance matrix

@dataclass
class LinearEkf:
    """ 
    ## Args:
    - **F** (Function[[np.ndarray], np.ndarray]): State Transition Matrix
    - **U** (Function[[np.ndarray], np.ndarray]): Input Matrix
    - **H** (Function[[np.ndarray], np.ndarray]): Observation Matrix
    - **Q** (np.ndarray): Process Noise Covariance. Typically a diagonal matrix where each elements
        the (expected) covariance of each element of the state
    - **R** (np.ndarray): Measurement Noise Covariance. Typically a diagonal matrix where each elements
        the (expected) covariance of each element of the measurement vector
    """
    F: np.ndarray
    B: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

@dataclass
class NonlinearEkf:
    """ 
    ## Args:
    - **get_f** (Function[[np.ndarray, np.ndarray], np.ndarray]): State Transition function
        [ get_f(state, input) --> next_state ]
    - **get_F** (Function[[np.ndarray], np.ndarray]): State Transition Covariance function
        [ get_F(state, input) --> F ]
    - **get_h** (Function[[np.ndarray, np.ndarray], np.ndarray]): Observation function
        (turns state into what theoretical measurements it would create in that state)
        [ get_h(state) --> meas_theoretical ]
    - **get_H** (Function[[np.ndarray], np.ndarray]): Observation Covariance function
        [ get_H(state) --> H ]
    - **Q** (np.ndarray): Process Noise Covariance. Typically a diagonal matrix where each elements
        the (expected) covariance of each element of the state
    - **R** (np.ndarray): Measurement Noise Covariance. Typically a diagonal matrix where each elements
        the (expected) covariance of each element of the measurement vector
    """
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    F: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray, np.ndarray], np.ndarray]
    H: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray


def run_linear_ekf(state: State, ekf: LinearEkf, u: np.ndarray, z: np.ndarray):
    F, B, H, Q, R = ekf.F, ekf.B, ekf.H, ekf.Q, ekf.R
    x, P = state.x, state.P
    dim = len(state.x)
    
    # Predict
    x_est = F @ x 
    if B is not None and u is not None: 
        x_est += B @ u
    P_est = F @ P @ F.T + Q

    # Update
    y_err = z - H @ x_est
    S = H @ P_est @ H.T + R
    K = P_est @ H.T @ np.linalg.inv(S)
    x_next = x_est + K @ y_err
    P_next = (np.eye(dim) - K@H) @ P_est @ (np.eye(dim) - K@H).T + K@R@K.T # Joseph form (not used)
    return State(x_next, P_next)

def run_nonlinear_ekf(state: State, ekf: NonlinearEkf, u: np.ndarray, z: np.ndarray):
    f, get_F, h, H_func, Q, R = ekf.f, ekf.F, ekf.h, ekf.H, ekf.Q, ekf.R
    x, P = state.x, state.P
    dim = len(state.x)
    
    # Predict
    x_est = f(x,u)
    F = get_F(x)
    P_est = F @ P @ F.T + Q

    # Update
    y_err = z - h(x_est)
    H = H_func(x)
    S = H @ P_est @ H.T + R
    K = P_est @ H.T @ np.linalg.inv(S)
    x_next = x_est + K @ y_err
    P_next = (np.eye(dim) - K@H) @ P_est @ (np.eye(dim) - K@H).T + K@R@K.T # Joseph form (not used)
    return State(x_next, P_next)