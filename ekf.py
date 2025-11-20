import numpy as np
from dataclasses import dataclass

@dataclass
class Ekf:
    F: np.ndarray
    B: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    dim: int

@dataclass
class State:
    x: np.ndarray
    P: np.ndarray
    dim: int


# def predict(x, u, F, B, P):
def run(state: State, ekf: Ekf, u: np.ndarray, z: np.ndarray):
    F, B, H, Q, R = ekf.F, ekf.B, ekf.H, ekf.Q, ekf.R
    x, P = state.x, state.P
    dim = state.dim # Should be the same as ekf dim

    # Predict
    x_est = F @ x + B @ u
    P_est = F @ P @ F.T + Q

    # Update
    y_err = z - H @ x_est
    S = H @ P_est @ H.T + R
    K = P_est @ H.T @ np.linalg.inv(S)
    x_next = x_est + K @ y_err
    P_next = (np.eye(dim) - K@H) @ P_est @ (np.eye(dim) - K@H).T + K@R@K.T

    return State(x_next, P_next, dim)

