from eki import eki_discrete
import helper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.pyplot import figure
import time
import config as cfg


def ensemble_convergence_1d():
    # parameters
    # h_list = [10 ** (-i) for i in range(4)]
    h_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    # plots
    param_string = f"Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU}, {cfg.SIGMA})"
    color_list = ['b', 'g', 'r', 'y', 'c', 'm', 'k']
    custom_lines = []
    
    # plot 1 (full time)
    fig_full, axs_full = plt.subplots(2, 2, figsize=(16, 12), dpi=80)
    fig_full.suptitle(f"Convergence results for 1D particles with G(u)=arctan(u) \n {param_string}")

    axs_full[0, 0].set_xlabel("time")
    axs_full[0, 0].set_ylabel("norm difference")
    axs_full[0, 0].set_title("norm convergence")
    axs_full[0, 0].set_xlim(0, cfg.T)

    axs_full[0, 1].set_xlabel("time")
    axs_full[0, 1].set_ylabel("u")
    axs_full[0, 1].set_title("ensemble convergence (random particles)")
    axs_full[0, 1].set_ylim(0, cfg.T)

    axs_full[1, 0].set_xlabel("time")
    axs_full[1, 0].set_ylabel("particle mean")
    axs_full[1, 0].set_title("mean value convergence")
    axs_full[1, 0].set_ylim(0, cfg.T)

    axs_full[1, 1].set_xlabel("time")
    axs_full[1, 1].set_ylabel("correlation")
    axs_full[1, 1].set_title("correlation convergence")
    axs_full[1, 1].set_xlim(0, cfg.T)
    
    # plot 2 (less time)
    fig_half, axs_half = plt.subplots(2, 2, figsize=(16, 12), dpi=80)
    fig_half.suptitle(f"Convergence results for 1D particles with G(u)=arctan(u) \n {param_string}")

    axs_half[0, 0].set_xlabel("time")
    axs_half[0, 0].set_ylabel("norm difference")
    axs_half[0, 0].set_title("norm convergence")
    axs_half[0, 0].set_xlim(cfg.T / 2, cfg.T)

    axs_half[0, 1].set_xlabel("time")
    axs_half[0, 1].set_ylabel("u")
    axs_half[0, 1].set_title("ensemble convergence (random particles)")
    axs_half[0, 1].set_ylim(cfg.T / 2, cfg.T)

    axs_half[1, 0].set_xlabel("time")
    axs_half[1, 0].set_ylabel("particle mean")
    axs_half[1, 0].set_title("mean value convergence")
    axs_half[1, 0].set_ylim(cfg.T / 2, cfg.T)

    axs_half[1, 1].set_xlabel("time")
    axs_half[1, 1].set_ylabel("correlation")
    axs_half[1, 1].set_title("correlation convergence")
    axs_half[1, 1].set_xlim(cfg.T / 2, cfg.T)

    for h in h_list:
        times, ensemble, corr = eki_discrete(h=h, num_sims=1)[0]

        # for plot legend
        custom_lines.append(Line2D([0], [0], color=color_list[h_list.index(h)], lw=4))

        # plot ensemble
        random_particles = random.choices(range(len(ensemble[0])), k=min(5, len(ensemble[0])))
        for i in random_particles:
            axs_full[0, 1].plot(times, [arr[i] for arr in ensemble], color_list[h_list.index(h)])
            axs_half[0, 1].plot(times[(len(times)//2):], [arr[i] for arr in ensemble[(len(ensemble)//2):]], color_list[h_list.index(h)])

        # plot norms & mean
        norms = []
        means = []
        for particles in ensemble:
            mean = helper.get_particle_mean(particles)
            means.append(mean[0])
            norm_sum = 0
            for i in range(len(particles)):
                norm_sum += np.linalg.norm(particles[i]-mean)**2
            norm_sum = 1/(len(particles)+1) * norm_sum
            norms.append(norm_sum[0])

        # plot norm
        axs_full[0, 0].plot(times, norms, color=color_list[h_list.index(h)])
        axs_half[0, 0].plot(times[(len(times)//2):], norms[(len(norms)//2):], color=color_list[h_list.index(h)])

        # plot mean
        axs_full[1, 0].plot(times, means, color=color_list[h_list.index(h)])
        axs_half[1, 0].plot(times[(len(times)//2):], means[(len(means)//2):], color=color_list[h_list.index(h)])

        # plot covariance/correlation
        axs_full[1, 1].plot(times, corr, color=color_list[h_list.index(h)])
        axs_half[1, 1].plot(times[(len(times)//2):], corr[(len(corr)//2):], color=color_list[h_list.index(h)])

    fig_full.legend(custom_lines, [f"h={h}" for h in h_list])
    fig_half.legend(custom_lines, [f"h={h}" for h in h_list])

    savetime = time.time()
    fig_full.savefig(f"plots/convergence_1d_full_{savetime}.png")
    fig_half.savefig(f"plots/convergence_1d_half_{savetime}.png")


def ensemble_convergence_2d():
    # parameters
    # h_list = [10 ** (-i) for i in range(4)]
    h_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    # plots
    param_string = f"Gamma={cfg.GAMMA}, y={cfg.Y[0]}, {cfg.PARTICLES} particles, u_0~N({cfg.MU}, {cfg.SIGMA})"
    color_list = ['b', 'g', 'r', 'y', 'c', 'm', 'k']
    custom_lines = []

    # plot 1 (full time)
    _, axs_full = plt.subplots(2, 2, figsize=(16, 12), dpi=80)
    fig_full = plt.figure(figsize=(16, 12), dpi=80)
    fig_full.suptitle(f"Convergence results for 1D particles with G(u)=arctan(u) \n {param_string}")
    axs_full[0, 0] = fig_full.add_subplot(2, 2, 1)
    axs_full[0, 1] = fig_full.add_subplot(2, 2, 2, projection='3d')
    axs_full[1, 0] = fig_full.add_subplot(2, 2, 3, projection='3d')
    axs_full[1, 1] = fig_full.add_subplot(2, 2, 4)

    axs_full[0, 0].set_xlabel("time")
    axs_full[0, 0].set_ylabel("norm difference")
    axs_full[0, 0].set_title("norm convergence")
    axs_full[0, 0].set_xlim(0, cfg.T)

    axs_full[0, 1].set_ylabel("time")
    axs_full[0, 1].set_xlabel("x")
    axs_full[0, 1].set_zlabel("y")
    axs_full[0, 1].set_title("ensemble convergence (random particles)")
    axs_full[0, 1].set_ylim(0, cfg.T)

    axs_full[1, 0].set_ylabel("time")
    axs_full[1, 0].set_xlabel("x")
    axs_full[1, 0].set_zlabel("y")
    axs_full[1, 0].set_title("mean value convergence")
    axs_full[1, 0].set_ylim(0, cfg.T)

    axs_full[1, 1].set_xlabel("time")
    axs_full[1, 1].set_ylabel("correlation norm")
    axs_full[1, 1].set_title("correlation norm convergence")
    axs_full[1, 1].set_xlim(0, cfg.T)

    # plot 2 (less time)
    _, axs_half = plt.subplots(2, 2, figsize=(16, 12), dpi=80)
    fig_half = plt.figure(figsize=(16, 12), dpi=80)
    fig_half.suptitle(f"Convergence results for 1D particles with G(u)=arctan(u) \n {param_string}")
    axs_half[0, 0] = fig_half.add_subplot(2, 2, 1)
    axs_half[0, 1] = fig_half.add_subplot(2, 2, 2, projection='3d')
    axs_half[1, 0] = fig_half.add_subplot(2, 2, 3, projection='3d')
    axs_half[1, 1] = fig_half.add_subplot(2, 2, 4)

    axs_half[0, 0].set_xlabel("time")
    axs_half[0, 0].set_ylabel("norm difference")
    axs_half[0, 0].set_title("norm convergence")
    axs_half[0, 0].set_xlim(cfg.T / 2, cfg.T)

    axs_half[0, 1].set_ylabel("time")
    axs_half[0, 1].set_xlabel("x")
    axs_half[0, 1].set_zlabel("y")
    axs_half[0, 1].set_title("ensemble convergence (random particles)")
    axs_half[0, 1].set_ylim(cfg.T/2, cfg.T)

    axs_half[1, 0].set_ylabel("time")
    axs_half[1, 0].set_xlabel("x")
    axs_half[1, 0].set_zlabel("y")
    axs_half[1, 0].set_title("mean value convergence")
    axs_half[1, 0].set_ylim(cfg.T/2, cfg.T)

    axs_half[1, 1].set_xlabel("time")
    axs_half[1, 1].set_ylabel("correlation norm")
    axs_half[1, 1].set_title("correlation norm convergence")
    axs_half[1, 1].set_xlim(cfg.T / 2, cfg.T)

    for h in h_list:
        times, ensemble, corr = eki_discrete(h=h, num_sims=1)[0]

        # for plot legend
        custom_lines.append(Line2D([0], [0], color=color_list[h_list.index(h)], lw=4))

        # plot ensemble
        random_particles = random.sample([*range(len(ensemble[0]))], k=min(5, len(ensemble[0])))
        for i in random_particles:
            axs_full[0, 1].plot([arr[i][0] for arr in ensemble], times, [arr[i][1] for arr in ensemble], color_list[h_list.index(h)])
            axs_half[0, 1].plot([arr[i][0] for arr in ensemble[(len(ensemble) // 2):]], times[(len(times) // 2):], [arr[i][1] for arr in ensemble[(len(ensemble) // 2):]],
                                color_list[h_list.index(h)])

        # plot norms & mean
        norms = []
        means = []
        for particles in ensemble:
            mean = helper.get_particle_mean(particles)
            means.append(mean)
            norm_sum = 0
            for i in range(len(particles)):
                norm_sum += np.linalg.norm(particles[i] - mean) ** 2
            norm_sum = 1 / (len(particles) + 1) * norm_sum
            norms.append(norm_sum)

        # plot norm
        axs_full[0, 0].plot(times, norms, color=color_list[h_list.index(h)])
        axs_half[0, 0].plot(times[(len(times) // 2):], norms[(len(norms) // 2):], color=color_list[h_list.index(h)])

        # plot mean
        axs_full[1, 0].plot([m[0] for m in means], times, [m[1] for m in means], color=color_list[h_list.index(h)])
        axs_half[1, 0].plot([m[0] for m in means][(len(means) // 2):], times[(len(times) // 2):], [m[1] for m in means][(len(means) // 2):], color=color_list[h_list.index(h)])

        # plot covariance/correlation norm
        axs_full[1, 1].plot(times, [np.linalg.norm(c) for c in corr], color=color_list[h_list.index(h)])
        axs_half[1, 1].plot(times[(len(times) // 2):], [np.linalg.norm(c) for c in corr][(len(corr) // 2):], color=color_list[h_list.index(h)])

    fig_full.legend(custom_lines, [f"h={h}" for h in h_list])
    fig_half.legend(custom_lines, [f"h={h}" for h in h_list])

    savetime = time.time()
    fig_full.savefig(f"plots/convergence_2d_full_{savetime}.png")
    fig_half.savefig(f"plots/convergence_2d_half_{savetime}.png")


if __name__ == "__main__":
    ensemble_convergence_2d()
