import os
import pickle
import random
import numpy as np
from riverswim import (
    RiverswimEnv6,
    RiverswimEnv6Kappa1,
    RiverswimEnv6Kappa2,
    RiverswimEnv6Kappa3,
    sigma_s_a_rs6,
    C_s_a_rs6,
)
from qlearning import qlearning, qlearning_es, qlearning_es_ucb, qlearning_ucb


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def run_exp1(
    algo,
    env_name,
    n_steps=300000,
    gamma=0.85,
    epsilon=0.1,
    c=0.55,
    n_steps_store=10,
    n_runs=50,
):
    """
    exp1: compute q-values for different types of q-learning algorithms 
    on different riverswim environments.
    """

    saved_name = "./results/riverswim6/{}/{}".format(env_name, algo)
    if not os.path.exists(saved_name):
        os.makedirs(saved_name)

    if env_name == "riverswimenv6":
        env = RiverswimEnv6()
    elif env_name == "riverswimenv6kappa1":
        env = RiverswimEnv6Kappa1()
    elif env_name == "riverswimenv6kappa2":
        env = RiverswimEnv6Kappa2()
    elif env_name == "riverswimenv6kappa3":
        env = RiverswimEnv6Kappa3()

    if algo == "qlearning":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning(env, n_steps, gamma, epsilon, n_steps_store)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_ucb":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_ucb(env, n_steps, gamma, epsilon, c, n_steps_store)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_es":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_es(
                env, sigma_s_a_rs6, C_s_a_rs6, n_steps, gamma, epsilon, n_steps_store
            )
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_es_ucb":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_es_ucb(
                env, sigma_s_a_rs6, C_s_a_rs6, n_steps, gamma, epsilon, c, n_steps_store
            )
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            print(algo)
            print(i_seed)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="qlearning")
    parser.add_argument("--env_name", type=str, default="riverswimenv6")
    parser.add_argument("--n_steps", type=int, default=300000)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.55)
    parser.add_argument("--n_steps_store", type=int, default=10)
    args = parser.parse_args()

    run_exp1(
        algo=args.algo,
        env_name=args.env_name,
        gamma=args.gamma,
        epsilon=args.epsilon,
        c=args.c,
        n_steps_store=args.n_steps_store,
    )
