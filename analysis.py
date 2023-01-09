import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from operator import add, sub
import decimal as d
import math
import time
import eki
import config as cfg
import helper
import pandas as pd
from typing import List
from tqdm import tqdm


def cauchy_analysis(sims: List = None):
    """
    Analysis of approximations via cauchy-sequences.
    Convergence of quadratic mean of the discrete EKI towards the continuous model is proven in theory.

    :param sims: Optional Simulation for multiple analysis functions on a single EKI
    """
    savetime = time.time()

    if sims is None:
        h_list = [1/2**i for i in range(10)]
        h_list.sort()
        print(h_list)
        sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
        interp_sims = helper.linear_interpolation(sims, h_list[0])
    else:
        interp_sims = sims
        h_list = list(interp_sims[0].keys())

    print("Computing cauchy errors")
    cauchy_errors = []
    for h in tqdm(h_list):
        row_list = [h]
        for h_tilde in h_list:
            particle_diff_norms = []
            for sim in interp_sims:
                particle_diff = sim[h].e[-1] - sim[h_tilde].e[-1]
                particle_diff_norms.append([np.linalg.norm(particle_diff[i])**2 for i in range(particle_diff.shape[0])])
            e_particle_diff_norms = []
            for p in range(cfg.PARTICLES):
                e_sum = 0
                for j in range(cfg.NUM_SIMS):
                    e_sum += particle_diff_norms[j][p]
                e_particle_diff_norms.append(e_sum / cfg.NUM_SIMS)
            mean_sum = 0
            for e_norm in e_particle_diff_norms:
                mean_sum += e_norm
            row_list.append(mean_sum / cfg.PARTICLES)
        cauchy_errors.append(row_list)
    cauchy_error_frame = pd.DataFrame(cauchy_errors, columns=['h']+h_list)
    #print(cauchy_error_frame)

    cauchy_error_frame.to_csv(f"plots/cauchy_errors_{savetime}.csv", sep=";")


def variance_analysis(sims: List = None):
    """
    Analysis of mean and variance of the ensemble.
    1. Analysis for each h over time.
    2. Analysis for all h at specific time.

    :param sims: Optional Simulation for multiple analysis functions on a single EKI
    """
    savetime = time.time()

    if sims is None:
        h_list = [1 / 2 ** i for i in range(10)]
        h_list.sort()
        print(h_list)
        sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
        interp_sims = helper.linear_interpolation(sims, h_list[0])
    else:
        interp_sims = sims
        h_list = list(interp_sims[0].keys())

    # plot
    param_string = f"{cfg.NUM_SIMS} simulations, Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU}, {cfg.SIGMA})"
    fig, axs = plt.subplots(1, 2, figsize=(16, 12), dpi=80)
    fig.suptitle(f"Variance analysis for 1D particles with G(u)={cfg.G_STRING} \n {param_string}")
    colormap = plt.get_cmap('viridis')
    color_list = colormap(np.linspace(0, 1, len(h_list)))
    custom_lines = []
    times = interp_sims[0][h_list[0]].t
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("mean estimation & standard deviation from mean")
    axs[1].set_title("Full")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("variance estimation")
    axs[1].set_title("Zoom")

    print("Computing mean & variance analysis")
    for h in tqdm(h_list):
        # for every h compute separately
        mus = []
        sigmas = []
        for sim in interp_sims:
            # create time series of mean and var in every simulation
            # put all series in one array of arrays
            sim_means = []
            sim_vars = []
            for ensemble in sim[h].e:
                # mean and variance in every ensemble
                mean = helper.get_particle_mean(ensemble)
                var = 0
                for row in ensemble:
                    var += (row-mean)**2
                var = var * 1/(ensemble.shape[0]-1)
                sim_means.append(mean)
                sim_vars.append(var)
            mus.append(sim_means)
            sigmas.append(sim_vars)
        # create array for mean over simulations
        mu = np.array(mus).mean(axis=0)
        sigma = np.array(sigmas).mean(axis=0)
        # convert array to list
        mu = [i[0] for i in mu.tolist()]
        sigma = [i[0] for i in sigma.tolist()]
        std_deviation = [math.sqrt(i) for i in sigma]
        # compute mu+sigma, mu-sigma
        mu_high = list(map(add, mu, std_deviation))
        mu_low = list(map(sub, mu, std_deviation))

        # for plot legend
        custom_lines.append(Line2D([0], [0], color=color_list[h_list.index(h)], lw=4))
        # plot mean
        axs[0].plot(times, mu, linestyle="-", color=color_list[h_list.index(h)])
        axs[0].plot(times, mu_low, linestyle="--", color=color_list[h_list.index(h)])
        axs[0].plot(times, mu_high, linestyle="--", color=color_list[h_list.index(h)])
        # plot variance
        axs[1].plot(times[-450:], mu[-450:], linestyle="-", color=color_list[h_list.index(h)])

    # save
    fig.legend(custom_lines, [f"h={h}" for h in h_list])
    fig.savefig(f"plots/variance_analysis_1D_{savetime}.png")


def error_analysis(sims: List = None):
    """
    Analysis of E(t,h) = \E 1/J \sum_j |u-mean|^2 at time T.
    For fixed t, E(t,h)->0 for h->0 .
    Additionally, E(h,t) <= Ct^\delta(h)

    :param sims: Optional Simulation for multiple analysis functions on a single EKI
    """
    savetime = time.time()

    if sims is None:
        h_list = [1 / 2 ** i for i in range(10)]
        h_list.sort()
        print(h_list)
        sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
        interp_sims = helper.linear_interpolation(sims, h_list[0])
    else:
        interp_sims = sims
        h_list = list(interp_sims[0].keys())

    print("Computing error analysis")
    errors = []
    for h in tqdm(h_list):
        expectance = 0
        for sim in interp_sims:
            mean = helper.get_particle_mean(sim[h].e[-1])
            particle_mean = 0
            for part in sim[h].e[-1]:
                particle_mean += np.linalg.norm(part-mean) ** 2
            particle_mean = particle_mean / cfg.PARTICLES
            expectance += particle_mean
        expectance = expectance / cfg.NUM_SIMS
        errors.append(expectance)

    # plot
    param_string = f"{cfg.NUM_SIMS} simulations, Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU},{cfg.SIGMA})"
    fig, axs = plt.subplots(1, 1, figsize=(16, 12), dpi=80)
    fig.suptitle(f"Error at t={cfg.T} for {cfg.DIMENSION}D particles with G(u)={cfg.G_STRING} \n {param_string}")
    # error
    axs.set_xlabel("h")
    axs.set_ylabel("error")
    axs.plot(h_list, errors, "-b")
    axs.invert_xaxis()
    # save
    fig.savefig(f"plots/error_analysis_{savetime}.png")


def histogramm_analysis(sims: List = None):
    """
        Histogram of simulation results to analyze the probability distribution apart from mean and variance.
        Only applies to the final time (i.e. the end result), or complexity of the plots would be too huge.
        Only applies to first and final step size h.

        :param sims: Optional Simulation for multiple analysis functions on a single EKI
        """
    savetime = time.time()

    if sims is None:
        h_list = [1 / 2 ** i for i in range(10)]
        h_list.sort()
        print(h_list)
        sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
        interp_sims = helper.linear_interpolation(sims, h_list[0])
    else:
        interp_sims = sims
        h_list = list(interp_sims[0].keys())

    # plots
    param_string = f"{cfg.NUM_SIMS} simulations, Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU}, {cfg.SIGMA})"
    fig, axs = plt.subplots(1, 2, figsize=(16, 12), dpi=80)
    fig.suptitle(f"Histogram analysis at T={cfg.T} and h={h_list[0]} with G(u)={cfg.G_STRING} \n {param_string}")
    axs[0].set_xlabel("u")
    axs[0].set_ylabel("#u")
    axs[0].set_title(f"particle means")
    axs[1].set_xlabel("u")
    axs[1].set_ylabel("#u")
    axs[1].set_title(f"all particles")

    print("Computing histogram analysis")
    all_particles = []
    means = []
    for sim in interp_sims:
        means.append(helper.get_particle_mean(sim[h_list[0]].e[-1]))
        for part in sim[h_list[0]].e[-1]:
            all_particles.append(part)
    # compute histograms
    means = [arr[0] for arr in means]
    axs[0].hist(means, bins=50)
    all_particles = [arr[0] for arr in all_particles]
    axs[1].hist(all_particles, bins=50)

    # save
    fig.savefig(f"plots/histogram_analysis_1D_{savetime}.png")


if __name__ == "__main__":
    # max acceptable range seems to be lower, because of some rounding errors
    h_list = [d.Decimal(1 / 2 ** i) for i in range(10)]
    h_list.sort()
    print(h_list)
    sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
    #interp_sims = helper.linear_interpolation(sims, h_list[0])

    #cauchy_analysis(interp_sims)
    #variance_analysis(interp_sims)
    #error_analysis(interp_sims)
    histogramm_analysis(sims)
