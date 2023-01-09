from typing import List, Tuple
from collections import namedtuple
import numpy as np
import eki
import config as cfg
from tqdm import tqdm


def linear_interpolation(sims: List, h_target: float) -> List:
    """
    Linear interpolation of the ensemble for missing times.
    ! h must be multiple of h_target !
    :param sims: List of simulations of the discrete EKI
    :param h_target: target step-size after interpolation
    :return: List of interpolated simulations
    """
    print("Starting Interpolation")
    new_sims = []
    for sim in tqdm(sims):
        if isinstance(sim, dict):
            new_h_dict = dict()
            for key in sim:
                interp_sim = interpolate_single_ensemble(sim[key], h_target)
                new_h_dict[key] = interp_sim
            new_sims.append(new_h_dict)
        else:
            interp_sim = interpolate_single_ensemble(sim, h_target)
            new_sims.append(interp_sim)

    return new_sims


def interpolate_single_ensemble(sim: Tuple, h_target: float) -> Tuple:
    times, ensemble, corr = sim
    #target_decimals = str(h_target)[::-1].find('.')
    h = times[1]
    # check if (h mod h_target == 0) but better here
    if not (h / h_target).as_integer_ratio()[1] == 1:
        raise ValueError("h-values do not fit. Simulation h must be multiple of h_target.")
    if h_target == h:
        return sim
    new_times = [times[0]]
    new_ensemble = [ensemble[0]]
    for t in range(len(times) - 1):
        # insert new times and ensembles (last time is already in larger mesh)
        insert_nr = int(h / h_target)
        for i in range(1, insert_nr):
            # calc new time
            new_time = times[t] + h_target * i
            new_times.append(new_time)
            # linear interpolation for new ensemble member
            scaling_factor = 1 / float(times[t + 1] - times[t])
            interp = ensemble[t] * float(times[t + 1] - new_time) + ensemble[t + 1] * float(new_time - times[t])
            new_ensemble.append(interp * scaling_factor)
        # insert bigger grid time
        new_times.append(times[t + 1])
        new_ensemble.append(ensemble[t + 1])

    new_corr = []
    for e in new_ensemble:
        new_corr.append(eki.covariance_matrix(e, cfg.G, "up"))

    # save interpolated "simulation"
    Sim = namedtuple('Sim', 't e c')
    return Sim(new_times, new_ensemble, new_corr)


def get_particle_mean(u: np.ndarray) -> np.ndarray:
    """
    Mean over all particles.
    :param u:
    :return:
    """
    return u.mean(axis=0)

