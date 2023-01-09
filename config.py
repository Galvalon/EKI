import latexify
import numpy as np

# global parameters for EKI
MU = 0.5
SIGMA = 1.
PARTICLES = 5
DIMENSION = 1
Y = np.ones(DIMENSION) * 0.5
GAMMA = np.eye(DIMENSION) * 0.001
T = 1
NUM_SIMS = 1000

# controllable random seed for experiments (None for no seed)
SEED = None

# always modify G_STRING when changing return of G (for correct documentation in outputs)
G_STRING = "0.5*u+1"


@latexify.with_latex
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
    return 0.5*u + 1

    # for nonlinear and non-invertible G (does not work! -> why?)
    """
    !!! This does only work for functions with a single solution !!!
    0.5 = sin(u)*u + GAMMA has two possible solutions u
    """
    #return np.multiply(np.sin(u), u)

    # nonlinear and invertible G (works!)
    #return np.arctan(u)

    # not covered in theory (seems to work!)
    #return np.power(u, 3)


