import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def f(t, x):
    return 1


def g(t, x):
    return 1


def euler_maruyama(f: Callable = f, g: Callable = g, X_0: float = 0.0, num_sims: int = 1, N: int = 100,
                   T: float = 1.0) -> np.ndarray:
    """
    SDE: dX_t = f(t,X_t) dt + g(t,X_t) dB_t
    :param f: drift function
    :param g: diffusion function
    :param X_0: initial value
    :param num_sims: number of simulations
    :param N: Number of points in partition
    :param T: end time (start time is always 0)
    :return:
    """

    # Initial value
    Y_0 = X_0

    # Time increments
    dt = T / N

    # Times
    t = np.arange(0, T, dt)

    # Brownian increments
    dB = np.zeros(t.size)
    dB[0] = 0

    # Brownian samples
    B = np.zeros(t.size)
    B[0] = 0

    # Simulated process
    X = np.zeros(t.size)
    X[0] = X_0

    # Approximated process
    Y = np.zeros(t.size)
    Y[0] = Y_0

    # Sample means across all simulations
    SX = np.zeros(t.size)
    SY = np.zeros(t.size)

    # Iterate
    for n in range(num_sims):
        for i in range(0, t.size):
            # Generate dB_t
            dB[i] = np.random.default_rng().normal(loc=0.0, scale=1.0)

            # Generate B_t
            B[i] = np.random.default_rng().normal(loc=0.0, scale=np.sqrt(t[i]))

            # Simulate (blue)
            X[i] = X_0 + i * dt + B[i]
            SX[i] = SX[i] + X[i] / num_sims

            # Approximate (green)
            Y[i] = Y[i - 1] + f(i * dt, Y[i - 1]) * dt + g(i * dt, Y[i - 1]) * np.sqrt(dt) * dB[i]
            SY[i] = SY[i] + Y[i] / num_sims
            #print(Y[i], SY[i], dB[i])

    # Plot
    plt.plot(t, SX, 'b')
    plt.plot(t, SY, 'g')
    # plt.plot(t, dB)
    plt.show()
    return SY


if __name__ == "__main__":
    for i in range(1, 100, 10):
        euler_maruyama(N=i, num_sims=50)
