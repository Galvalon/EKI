import numpy as np

# global parameters for EKI
MU = 3.
SIGMA = 1.
PARTICLES = 5
DIMENSION = 1
Y = np.ones(DIMENSION) * 1.0
GAMMA = np.eye(DIMENSION) * 0.001
T = 1
NUM_SIMS = 1000

# controllable random seed for experiments (None for no seed)
SEED = None

# solution dict for histograms (only creates plots if G_STRING is a key here)
basic_solutions = {'0.5u+1': [-1], '2u-0.1': [0.3], 'sin(u)u': [-2.97, -0.74, 0.74, 2.97], 'arctan(u)': [0.55], 'u^3': [0.79], '|u|': [-0.5, 0.5], 'floor(u)': True}

# always modify G_STRING when changing return of G (for correct documentation in outputs)
G_STRING = "floor(u)"


def G(u: np.ndarray) -> np.ndarray:
    """
    Forward map. (Matrices need transposing)
    :param u: array-like
    :return: array-like
    """
    # for linear G 2D (works!)
    #A = np.eye(DIMENSION)
    #A[-1, -1] = 0.01
    # formula: Au, problem: u is transposed -> formula becomes (A * u.T).T, faster solution: u * A.T = (A + u.T).T
    #return np.matmul(u, A.transpose())

    # for linear G 1D (works!)
    #return 0.5*u + 1
    #return 2*u - 0.1

    # for nonlinear and non-invertible G (bad results, but expected from theory)
    #return np.multiply(np.sin(u), u)

    # nonlinear and invertible G
    #return np.arctan(u)

    # not covered in all theory (not lin. bounded)
    #return np.power(u, 3)

    # step-function (not covered in theory as not diff.)
    return np.floor(u)

    # not diff. (not covered in theory as not diff.)
    #return np.abs(u)
