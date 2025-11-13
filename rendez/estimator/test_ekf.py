from .ekf import get_F, get_H, get_B
import numpy as np

# These are not exhaustive tests
def test_F():
    n = 15

    X = np.ones(6)
    u = np.ones(3)

    F = get_F(X, u, n)

    print(F)

    assert np.array_equal(F, np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n*n, 0, 0, 0, 2*n, 0],
        [0, 0, 0,  -2*n, 0, 0],
        [0, 0,  -n*n, 0, 0, 0]
    ]))

def test_B():

    dt = 0.1

    B = get_B(dt)

    print(B)

    assert np.array_equal(B, np.array([
        [0.5*dt*dt, 0, 0],
        [0, 0.5*dt*dt, 0],
        [0, 0, 0.5*dt*dt],
        [dt,0,0],
        [0,dt,0],
        [0,0,dt]
    ]))

def test_H():

    H = get_H()

    print(H)

    assert np.array_equal(H, np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]))

    


