import latexify
import numpy as np

# global parameters for EKI
MU = 0.
SIGMA = 1.
PARTICLES = 2
DIMENSION = 1
Y = np.ones(DIMENSION)
GAMMA = np.eye(DIMENSION) * 0.001
T = 1.0
NUM_SIMS = 1000

# controllable random seed for experiments (None for no seed)
SEED = None


@latexify.with_latex
def G(u: np.ndarray) -> np.ndarray:
    """
    Forward map. (Matrices need transposing)
    :param u: array-like
    :return: array-like
    """
    # for linear G
    #A = np.diag(1)
    #A[-1, -1] = 0.01
    # formula: Au, problem: u is transposed -> formula becomes (A * u.T).T, faster solution: u * A.T = (A + u.T).T
    #return np.matmul(u, A.transpose())

    # for nonlinear and non-invertible G
    #return np.

    # nonlinear and invertible G
    return np.arctan(u)


