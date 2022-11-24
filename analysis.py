import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
from operator import add, sub
import math
import time
import eki
import config as cfg
import helper
import pandas as pd


def cauchy_analysis():
    """
    Analysis of approximations via cauchy-sequences.
    Convergence of quadratic mean of the discrete EKI towards the continuous model is proven in theory.

    """
    h_list = [1/2**i for i in range(10)]
    h_list.sort()
    print(h_list)
    sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
    interp_sims = helper.linear_interpolation(sims, h_list[0])

    cauchy_errors = []
    for h in h_list:
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
    print(cauchy_error_frame)

    cauchy_error_frame.to_csv("Cauchy_Errors.csv", sep=";")


def variance_analysis():
    """
    Analysis of mean and variance of the ensemble.
    1. Analysis for each h over time.
    2. Analysis for all h at specific time.
    """
    savetime = time.time()

    h_list = [1 / 2 ** i for i in range(10)]
    h_list.sort()
    print(h_list)
    sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
    interp_sims = helper.linear_interpolation(sims, h_list[0])

    for h in h_list:
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
                mean = eki.get_particle_mean(ensemble)
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

        # plot
        param_string = f"Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU}, {cfg.SIGMA})"
        fig, axs = plt.subplots(1, 2, figsize=(16, 12), dpi=80)
        fig.suptitle(f"Variance analysis for 1D particles with G(u)=arctan(u) & h={h} \n {param_string}")
        times = interp_sims[0][h_list[0]].t
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("mean & variance")
        axs[1].set_title("Full")
        axs[0].plot(times, mu, "-b")
        axs[0].plot(times, mu_low, "-g")
        axs[0].plot(times, mu_high, "-g")
        # plot without steep curve at the start
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("mean & variance")
        axs[1].set_title("Zoom")
        axs[1].plot(times[-450:], mu[-450:], "-b")
        axs[1].plot(times[-450:], mu_low[-450:], "-g")
        axs[1].plot(times[-450:], mu_high[-450:], "-g")
        # save
        fig.savefig(f"plots/variance_analysis_1D_h={h}_{savetime}.png")


def error_analysis():
    """
    Analysis of E(t,h) = \E 1/J \sum_j (u-mean)^2 at time T.
    For fixed t, E(t,h)->0 for h->0 .
    Additionally, E(h,t) <= Ct^\delta(h)
    """
    savetime = time.time()

    h_list = [1 / 2 ** i for i in range(10)]
    h_list.sort()
    print(h_list)
    sims = eki.eki_discrete_fixed_randomness(h_list=h_list)
    interp_sims = helper.linear_interpolation(sims, h_list[0])

    errors = []
    for h in h_list:
        expectance = 0
        for sim in interp_sims:
            mean = eki.get_particle_mean(sim[h].e[-1])
            particle_mean = 0
            for part in sim[h].e[-1]:
                particle_mean += np.linalg.norm(part-mean) ** 2
            particle_mean = particle_mean / cfg.PARTICLES
            expectance += particle_mean
        expectance = expectance / cfg.NUM_SIMS
        errors.append(expectance)

    # plot
    param_string = f"Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU},{cfg.SIGMA})"
    fig, axs = plt.subplots(1, 2, figsize=(16, 12), dpi=80)
    fig.suptitle(f"Error at t={cfg.T} for 1D particles with G(u)=arctan(u) \n {param_string}")
    # error
    axs[0].set_xlabel("h")
    axs[0].set_ylabel("error")
    axs[0].plot(h_list, errors, "-b")
    axs[0].invert_xaxis()
    # bound
    axs[1].set_xlabel("h")
    axs[1].set_ylabel("log(error)")
    axs[1].set_yscale('log')
    axs[1].plot(h_list, errors, "-b")
    axs[1].invert_xaxis()
    # save
    fig.savefig(f"plots/error_analysis_{savetime}.png")


if __name__ == "__main__":
    error_analysis()
