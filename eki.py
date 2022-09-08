import numpy as np
from typing import Callable, List, Tuple
from scipy.linalg import fractional_matrix_power
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from profiling import profileit


def G(u: np.ndarray) -> np.ndarray:
    """
    Forward map. Rewrite EKI for faster computation (np.apply_along_axis is slow for many rows/cols).
    :param u: vector
    :return: vector
    """
    return u


def initial_ensemble(num_particles: int, dim: int, mu: float = 0., sigma: float = 1.):
    """
    Define an initial ensemble
    :param num_particles:
    :param dim:
    :param mu:
    :param sigma:
    :return:
    """
    return np.random.default_rng().normal(mu, sigma, (num_particles, dim))


def correlation_matrix(u: np.ndarray, G: Callable = None, idx: str = "up") -> np.ndarray:
    """
    covariance/correlation matrix computation
    :param u: ensemble (every row is ensemble member)
    :param G: forward map (on ensemble members, i.e. row of u)
    :param idx: "uu", "up" or "pp"
    :return: matrix
    """
    if G is None:
        print("No function G supplied, computing C^uu")
    if idx == "uu" or G is None:
        u_scaled = u - u.mean(axis=0)
        # 1/J sum_j (u_j-mean)^T (u_j-mean)
        return 1 / u_scaled.shape[0] * np.matmul(u_scaled.transpose(), u_scaled)
    elif idx == "up":
        u_scaled = u - u.mean(axis=0)
        gu = np.apply_along_axis(G, 1, u)
        gu_scaled = gu - gu.mean(axis=0)
        return 1 / u_scaled.shape[0] * np.matmul(u_scaled.transpose(), gu_scaled)
    elif idx == "pp":
        gu = np.apply_along_axis(G, 1, u)
        gu_scaled = gu - gu.mean(axis=0)
        return 1 / gu_scaled.shape[0] * np.matmul(gu_scaled.transpose(), gu_scaled)
    else:
        print("Only 'uu', 'up' and 'pp' allowed.")


@profileit
def eki_discrete(gamma: np.ndarray, h: float, y: np.ndarray, G: Callable = G, N: int = None, num_sims:int = 1, init_shape: Tuple[int, int] = None, u_0: np.ndarray = None) -> np.ndarray:
    """
    Compute the EKI in discrete time of inverse problem y=G(u)+noise.
    ! All matrix multiplications are transposed as ensemble vectors are rows of u !
    :param gamma: covariance matrix
    :param h: step size (dt)
    :param y:
    :param G:
    :param N: optional step amount
    :param num_sims:
    :param init_shape: shape for initial ensemble =(ensemble_size, dimension)
    :param u_0: optional initial ensemble
    :return: resulting ensemble, shape =(num_sims, ensemble_size, dimension)
    """
    if N is None:
        N = int(1/h)
    if init_shape is not None:
        y_matrix = np.tile(y, (init_shape[0], 1))
    elif u_0 is not None:
        y_matrix = np.tile(y, (u_0.shape[0], 1))
    else:
        print("Init unknown. Aborting.")
        exit(1)
    u_sims = []
    for _ in tqdm(range(num_sims)):
        if init_shape is not None:
            u_0 = initial_ensemble(num_particles=init_shape[0], dim=init_shape[1])
        u_new = u_0
        for _ in range(N):
            # save old u
            u_old = u_new
            # compute G(u) for ensemble
            gu = np.apply_along_axis(G, 1, u_old)
            # create random vectors ~N(0,Id) for ensemble size u.shape[0]
            random_matrix = np.random.default_rng().multivariate_normal(np.zeros(u_old.shape[1]), np.eye(u_old.shape[1]),
                                                                        u_old.shape[0])
            scaled_random_matrix = np.matmul(random_matrix, fractional_matrix_power(gamma, 0.5))
            # compute matrices
            bracket_matrix = h * correlation_matrix(u_old, G, "pp") + gamma
            prematrix = np.matmul(correlation_matrix(u_old, G, "up"), np.linalg.inv(bracket_matrix))
            #print(y_matrix.shape, gu.shape, prematrix.shape)
            #print(scaled_random_matrix.shape, prematrix.shape)
            u_new = u_old + h * np.matmul(y_matrix - gu, prematrix) + np.sqrt(h) * np.matmul(scaled_random_matrix, prematrix)
        u_sims.append(u_new)
    return np.array(u_sims)


def get_particle_mean(u: np.ndarray) -> np.ndarray:
    """
    Mean over all particles.
    :param u:
    :return:
    """
    if len(u.shape) == 2:
        return u.mean(axis=0)
    elif len(u.shape) == 3:
        return u.mean(axis=1)


def get_moments(u: np.ndarray, max_moment: int = 2) -> Tuple:
    """
    Compute moment.
    :param u:
    :param max_moment: highest moment to compute
    :return:
    """
    moments = stats.describe(u, ddof=0)
    return moments.mean, moments.variance
    #return u.mean(axis=0), u.var(axis=0)


if __name__ == "__main__":
    #u_0 = initial_ensemble(100, 1)
    #print("Initial ensemble")
    #print(u_0.shape)
    #print(u_0.mean(axis=0))
    ensemble_size = 10
    dim = 1
    gamma = np.eye(dim) * 0.1
    y = np.ones(dim)
    u_eki = eki_discrete(gamma=gamma, h=0.01, init_shape=(10, 1), y=y, num_sims=10)
    print("EKI result shape")
    print(u_eki.shape)
    u = get_particle_mean(u_eki)
    print("Particle mean")
    print(u.shape)
    print(u)
    mu, sigma = get_moments(u, 2)
    print("Moments")
    print(mu)
    print(sigma)
    print(correlation_matrix(u))
