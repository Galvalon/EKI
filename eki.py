import numpy as np
from typing import Callable, List, Tuple, Dict, Union
from scipy.linalg import fractional_matrix_power
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from profiling import profileit
from collections import namedtuple
import config as cfg


def initial_ensemble(num_particles: int, dim: int, mu: float, sigma: float, rng=None) -> np.ndarray:
    """
    Define an initial ensemble
    :param num_particles:
    :param dim:
    :param mu:
    :param sigma:
    :param rng: random generator
    :return: array with particles as rows
    """
    return rng.normal(mu, sigma, (num_particles, dim))


def correlation_matrix(u: np.ndarray, G: Union[Callable, np.ndarray], idx: str = "up") -> np.ndarray:
    """
    covariance/correlation matrix computation
    :param u: ensemble (every row is ensemble member)
    :param G: G or G(u)
    :param idx: "uu", "up" or "pp"
    :return: matrix
    """
    # compute G if necessary
    if G is None:
        print("No function G supplied, computing C^uu")
    elif type(G) == np.ndarray:
        gu = G
    elif type(G) != np.ndarray:
        gu = G(u)
    # compute correlation
    if idx == "uu" or G is None:
        u_scaled = u - u.mean(axis=0)
        # 1/J sum_j (u_j-mean)^T (u_j-mean)
        return 1 / u_scaled.shape[0] * np.matmul(u_scaled.transpose(), u_scaled)
    elif idx == "up":
        u_scaled = u - u.mean(axis=0)
        gu_scaled = gu - gu.mean(axis=0)
        return 1 / u_scaled.shape[0] * np.matmul(u_scaled.transpose(), gu_scaled)
    elif idx == "pp":
        gu_scaled = gu - gu.mean(axis=0)
        return 1 / gu_scaled.shape[0] * np.matmul(gu_scaled.transpose(), gu_scaled)
    else:
        print("Only 'uu', 'up' and 'pp' allowed.")


@profileit
def eki_discrete(h_list: List[float]) -> List:
    """
    Compute the EKI in discrete time of inverse problem y=G(u)+noise.
    ! All matrix multiplications are transposed as ensemble vectors are rows of u !
    Configuration is always taken from config.py.
    :param h_list: step sizes
    :return: resulting simulations
    """
    # create random generator
    rng = np.random.default_rng(seed=cfg.SEED)
    # init
    y_matrix = np.tile(cfg.Y, (cfg.PARTICLES, 1))
    sims = []
    # decimals for times
    target_decimals = str(h_list[0])[::-1].find('.')
    for _ in tqdm(range(cfg.NUM_SIMS)):
        h_sims = dict()
        for h in h_list:
            N = int(cfg.T / h)
            # return vars
            Sim = namedtuple('Sim', 't e c')
            corr = []
            ensembles = []
            times = []
            u_0 = initial_ensemble(num_particles=cfg.PARTICLES, dim=cfg.DIMENSION, mu=cfg.MU, sigma=cfg.SIGMA, rng=rng)
            u_new = u_0
            ensembles.append(u_new)
            for n in range(N):
                times.append(round(n*h), target_decimals)
                # save old u
                u_old = u_new
                # compute G(u) for ensemble
                gu = cfg.G(u_old)
                # save correlation
                corr.append(correlation_matrix(u_old, gu, "up"))
                # create random vectors ~N(0,Id) for ensemble size u.shape[0]
                random_matrix = rng.multivariate_normal(np.zeros(u_old.shape[1]), np.eye(u_old.shape[1]),
                                                                            u_old.shape[0])
                scaled_random_matrix = np.matmul(random_matrix, fractional_matrix_power(cfg.GAMMA, 0.5))
                # compute matrices
                bracket_matrix = h * correlation_matrix(u_old, gu, "pp") + cfg.GAMMA
                prematrix = np.matmul(correlation_matrix(u_old, gu, "up"), np.linalg.inv(bracket_matrix))

                u_new = u_old + h * np.matmul(y_matrix - gu, prematrix) + np.sqrt(h) * np.matmul(scaled_random_matrix, prematrix)

                ensembles.append(u_new)

            times.append(N*h)
            corr.append(correlation_matrix(u_new, cfg.G, "up"))
            # pack for return
            h_sims[h] = Sim(times, ensembles, corr)
        # pack sims together
        sims.append(h_sims)
    return sims


@profileit
def eki_discrete_fixed_randomness(h_list: List[float]) -> List:
    """
    Compute the EKI in discrete time of inverse problem y=G(u)+noise.
    For every simulation, the randomness at a specific time is fixed (taken from the randomness of the smallest time steps)
    ! All matrix multiplications are transposed as ensemble vectors are rows of u !
    Configuration is always taken from config.py.
    :param h_list: step sizes (must be multiples of smallest element)
    :return: resulting ensembles as list of dicts
    """
    # create random generator
    rng = np.random.default_rng(seed=cfg.SEED)
    # init
    y_matrix = np.tile(cfg.Y, (cfg.PARTICLES, 1))
    sims = []
    # start with smallest h
    h_list.sort()
    # decimals for times
    target_decimals = str(h_list[0])[::-1].find('.')
    for _ in tqdm(range(cfg.NUM_SIMS)):
        u_0 = initial_ensemble(num_particles=cfg.PARTICLES, dim=cfg.DIMENSION, mu=cfg.MU, sigma=cfg.SIGMA, rng=rng)
        random_dW = dict()
        for n in range(int(cfg.T / h_list[0])):
            random_dW[round(n*h_list[0], target_decimals)] = rng.multivariate_normal(np.zeros(u_0.shape[1]), np.eye(u_0.shape[1]),
                                                                            u_0.shape[0])
        h_sims = dict()
        for h in h_list:
            u_new = u_0
            N = int(cfg.T / h)
            # return vars
            Sim = namedtuple('Sim', 't e c')
            corr = []
            ensembles = []
            times = []
            ensembles.append(u_new)
            for n in range(N):
                times.append(round(n*h, target_decimals))
                # save old u
                u_old = u_new
                # compute G(u) for ensemble
                gu = cfg.G(u_old)
                # save correlation
                corr.append(correlation_matrix(u_old, gu, "up"))
                # create random vectors ~N(0,Id) for ensemble size u.shape[0]
                random_matrix = random_dW[round(n*h, target_decimals)]
                scaled_random_matrix = np.matmul(random_matrix, fractional_matrix_power(cfg.GAMMA, 0.5))
                # compute matrices
                bracket_matrix = h * correlation_matrix(u_old, gu, "pp") + cfg.GAMMA
                prematrix = np.matmul(correlation_matrix(u_old, gu, "up"), np.linalg.inv(bracket_matrix))

                u_new = u_old + h * np.matmul(y_matrix - gu, prematrix) + np.sqrt(h) * np.matmul(scaled_random_matrix, prematrix)

                ensembles.append(u_new)

            times.append(N*h)
            corr.append(correlation_matrix(u_new, cfg.G, "up"))
            # pack for return
            h_sims[h] = Sim(times, ensembles, corr)
        # pack sims together
        sims.append(h_sims)
    return sims


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


def get_moments(u: np.ndarray) -> Tuple:
    """
    Compute moment.
    :param u:
    :return:
    """
    moments = stats.describe(u, ddof=0)
    return moments.mean, moments.variance


if __name__ == "__main__":
    h = 0.01

    times, ensemble, corr = eki_discrete(h=h)[0]
