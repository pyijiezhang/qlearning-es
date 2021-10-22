import numpy as np
import copy
from utils import make_epsilon_greedy_policy


def qlearning(env, n_steps=300000, gamma=0.85, epsilon=0.1, n_steps_store=10):

    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    count_s_a = {}
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count (s,a)
        if (s, a) in count_s_a:
            count_s_a[(s, a)] += 1
        else:
            count_s_a[(s, a)] = 1

        # update learning rate
        alpha_t = 10 / (count_s_a[(s, a)] + 1)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + gamma * max_q - Q[s][a])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_ucb(
    env, n_steps=300000, gamma=0.85, epsilon=0.1, c=0.55, n_steps_store=10
):

    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    Q_hat = {}
    for s in range(env.nS):
        Q[s] = [1 / (1 - gamma)] * env.nA
        Q_hat[s] = [1 / (1 - gamma)] * env.nA

    policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

    H = 1 / (1 - gamma)
    count_s_a = {}
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count occurence of (s,a)
        if (s, a) in count_s_a:
            count_s_a[(s, a)] += 1
        else:
            count_s_a[(s, a)] = 1
        k = count_s_a[(s, a)]
        b_k = c * np.sqrt(H / k)
        alpha_t = (H + 1) / (H + k)

        max_q = np.max(Q_hat[s_prime])
        Q[s][a] += alpha_t * (r + b_k + gamma * max_q - Q[s][a])
        Q_hat[s][a] = np.min([Q[s][a], Q_hat[s][a]])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_es(
    env, sigma_s_a, C_s_a, n_steps=300000, gamma=0.85, epsilon=0.1, n_steps_store=10
):
    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # init es count
    count_C_s_a = {}
    for c in range(len(C_s_a)):
        count_C_s_a[c] = 0

    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count es and update learning rate
        for c in range(len(C_s_a)):
            if (s, a) in C_s_a[c]:
                count_C_s_a[c] += 1
                alpha_t = 10 / (count_C_s_a[c] + 1)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + gamma * max_q - Q[s][a])

        for pairs_es in sigma_s_a[s, a, s_prime]:
            s_es, a_es = pairs_es[0]
            s_prime_es = pairs_es[1]
            r_es = pairs_es[2]
            max_q_es = np.max(Q[s_prime_es])
            Q[s_es][a_es] += alpha_t * (r_es + gamma * max_q_es - Q[s_es][a_es])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_es_ucb(
    env,
    sigma_s_a,
    C_s_a,
    n_steps=300000,
    gamma=0.85,
    epsilon=0.1,
    c=0.55,
    n_steps_store=10,
):
    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    Q_hat = {}
    for s in range(env.nS):
        Q[s] = [1 / (1 - gamma)] * env.nA
        Q_hat[s] = [1 / (1 - gamma)] * env.nA

    policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

    # init es count
    count_C_s_a = {}
    for c in range(len(C_s_a)):
        count_C_s_a[c] = 0

    H = 1 / (1 - gamma)
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count es and update learning rate
        for c in range(len(C_s_a)):
            if (s, a) in C_s_a[c]:
                count_C_s_a[c] += 1
                k = count_C_s_a[c]
                b_k = c * np.sqrt(H / k)
                alpha_t = (H + 1) / (H + k)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + b_k + gamma * max_q - Q[s][a])
        Q_hat[s][a] = np.min([Q[s][a], Q_hat[s][a]])

        for pairs_es in sigma_s_a[s, a, s_prime]:
            s_es, a_es = pairs_es[0]
            s_prime_es = pairs_es[1]
            r_es = pairs_es[2]
            max_q_es = np.max(Q[s_prime_es])
            Q[s_es][a_es] += alpha_t * (r_es + b_k + gamma * max_q_es - Q[s_es][a_es])
            Q_hat[s_es][a_es] = np.min([Q[s_es][a_es], Q_hat[s_es][a_es]])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all

