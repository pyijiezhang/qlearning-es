import numpy as np
import matplotlib.pyplot as plt


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        idx_max = np.argmax(Q[observation])
        probabilities = np.ones(nA) * (epsilon / nA)
        probabilities[idx_max] += 1.0 - epsilon
        return probabilities

    return policy_fn


def make_uniform_policy(nA):
    def policy_fn():
        probabilities = np.ones(nA) * (1.0 / nA)
        return probabilities

    return policy_fn


def plot_CI(results, label, n_steps):
    """ plot confidence intervals """
    results_mean = np.mean(results, axis=0)
    results_sde = 1.95 * np.std(results, axis=0) / np.sqrt(results.shape[0])
    plt.grid()
    plt.plot(
        np.linspace(0.0, n_steps, num=results.shape[1]), results_mean, label=label,
    )

    plt.fill_between(
        np.linspace(0.0, n_steps, num=results.shape[1]),
        results_mean + results_sde,
        results_mean - results_sde,
        alpha=0.3,
    )

