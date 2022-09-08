from eki import eki_discrete, get_particle_mean, get_moments
import numpy as np
import matplotlib.pyplot as plt


h_list = [10**(-i) for i in range(5)]
mu_list = []
sigma_list = []
max_particle = []
min_particle = []

for h in h_list:
    print(h)
    gamma = np.eye(1) * 0.1
    y = np.ones(1)
    u_eki = eki_discrete(gamma=gamma, h=h, init_shape=(100, 1), y=y, num_sims=100)
    # only for 1D particles
    max_particle.append(np.amax(u_eki))
    min_particle.append(np.amin(u_eki))

    u = get_particle_mean(u_eki)
    mu, sigma = get_moments(u, max_moment=2)
    mu_list.append(mu)
    sigma_list.append(sigma)

plt.plot(h_list, mu_list, 'b', label="mean")
plt.plot(h_list, max_particle, 'g', label="max")
plt.plot(h_list, min_particle, 'r', label="min")
#plt.plot(h_list, sigma_list, 'g')
plt.xscale("log")
plt.show()
plt.savefig("plots.png")
