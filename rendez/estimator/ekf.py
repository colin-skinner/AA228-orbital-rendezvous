import numpy as np

"""
Translation Kalman Filter

Initialization:
- R0, Q0, P0
- Initial (predicted) state
    - (x, y, z, sensor bias?)

Inputs:
- Noisy state
- Noisy absolute measurements

Outputs:
- Position estimate

"""


##########################################################################################
#               Functions to get F, B, and H
##########################################################################################

def get_F(X: np.ndarray, input: np.ndarray, n: float):
    """State transition matrix (Built for nonlinear, but currently linear)"""

    F = np.zeros((6,6))

    # Top right is identity matrix
    F[0:3, 3:6] = np.eye(3)
    F[3,0] = 3*n*n
    F[3,4] = 2*n
    F[4,3] = -2*n
    F[5,2] = -n*n

    return F

def get_B(dt: float):
    """Control input model maps acceleration to x,v """
    B = np.zeros((6,3))

    # accel to pos
    B[0,0] = 0.5*dt*dt
    B[1,1] = 0.5*dt*dt
    B[2,2] = 0.5*dt*dt

    # accel to velocity
    B[3,0] = dt
    B[4,1] = dt
    B[5,2] = dt

    return B

def get_H():
    """Maps relative position measurements to state"""

    H = np.zeros((6,3))
    H[:3, :3] = np.eye(3)

    return H

##########################################################################################
#               Prediction / Models
##########################################################################################


def prediction_step(state: np.ndarray, dt: float, u_input: np.ndarray, n: float, P: np.ndarray, Q: np.ndarray):
    """Propagates truth state vector

    Args:
        X (np.ndarray): Current state vector in target L frame (6,) [x, y, z, vx, vy, vz] 
        dt (float): Timestep size (sec)
        u_input (np.ndarray): Input vector (3,) [ux, uy, uz]
    """

    F = get_F(state, u_input, n)
    B = get_B(dt)

    state_predicted = state + dt * F @ state + B @ u_input
    next_P = F @ P @ F.T + Q

    return state_predicted, next_P

def measurement_model(state: np.ndarray):
    """Current measurements come from relative position in chaser frame (eventually GPS position)"""

    H = get_H()

    return H @ state


##########################################################################################
#               Update
##########################################################################################
    
def update_step(state_est: np.ndarray, z_meas: np.ndarray, P: np.ndarray, R: np.ndarray):

    # Measurement error
    y_err = z_meas - measurement_model(state_est)

    # Kalman Gain
    H = get_H()
    S = R + H @ P @ H.T
    K = P @ H.T @ np.linalg.inv(S)

    # Update
    state_updated = state_est + K @ y_err

    # Covariance propogation
    next_P = (np.eye(6) - K @ H) @ P

    return state_updated, next_P

##########################################################################################
#               Full EKF!
##########################################################################################

def ekf_step(
        state: np.ndarray, 
        u_input: np.ndarray, 
        z_meas: np.ndarray,
        dt: float, 
        n: float, 
        P: np.ndarray, 
        Q: np.ndarray,
        R: np.ndarray):
    
    state_pre, P_pre = prediction_step(state, dt, u_input, n, P, Q)
    state_post, P_post = update_step(state_pre, z_meas, P_pre, R)

    return state_post, P_post


if __name__ == "__main__":

    np.set_printoptions(precision=3)

    X = np.array([0,1,2,3,4,5])
    U = np.array([1,2,3])
    dt = 0.1

    mu_earth = 3.986e14 # m3/s2
    a = 6793137 # m (orbital radius)

    n = np.sqrt(mu_earth/(a**3))

    print(get_F(X, U, n))

    print(get_B(1))