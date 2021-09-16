import numpy as np
import copy


def policy_eval(policy, env, gamma=0.85, theta=0.001):
    """
    evaluate a policy given an environment and a full description of the environment's dynamics
    
    args:
        policy:[S,A] shaped matrix representing the policy
        env:
            openai env. env.P represents the transition probabilities of the environment
            env.P[s][a] is a list of transition tuples (prob,next_state,reward,done)
            env.nS is a numbear of states in the environment
            env.nA is a number of actions in the environment
        theta:stop evaluation once our value function change is less than theta for all states
        gamma:discount factor
    
    returns:
        a vector of length env.nS representing the value function
    """
    # start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for state in env.P:
            v_old = V[state]
            v_new = 0.0
            for action in env.P[state]:
                q_value = 0.0
                for transition_tuple in env.P[state][action]:
                    prob, next_state, reward, done = transition_tuple
                    q_value += prob * (reward + gamma * V[next_state])
                v_new += policy[state][action] * q_value
            V[state] = v_new
            delta = np.maximum(delta, np.abs(v_new - v_old))
        if delta < theta:
            break
    return np.array(V)


def policy_iteration(env, gamma=0.85):
    """
    policy iteration iteratively evaluates and improves a policy
    until an optimal policy is found
    
    args:
        env:openai envrionment
        policy_eval_fn:policy evaluation function that takes 3 arguments:
            policy,env,gamma
        gamma:discount factor
        
    returns:
        a tuple (policy,V):
            policy is the optimal policy,a matrix of shape [S,A] where each state s
            contains a valid probability distribution over actions
            V is the value function for the optimal policy
    """
    # start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        V = policy_eval(policy, env, gamma)

        policy_stable = True
        for state in env.P:
            old_action = copy.deepcopy(policy[state])

            new_policies = np.zeros(env.nA)
            for action in env.P[state]:
                q_value = 0.0
                for transition_tuple in env.P[state][action]:
                    prob, next_state, reward, done = transition_tuple
                    q_value += prob * (reward + gamma * V[next_state])
                new_policies[action] = q_value

            policy[state] *= 0.0
            policy[state][np.argmax(new_policies)] = 1
            if np.any(policy[state] != old_action):
                policy_stable = False
        if policy_stable:
            break

    return policy, V


def value_iteration(env, theta=0.001, gamma=0.85):
    """
    value iteration
    
    args:
        env:
            openai env. env.P represents the transition probabilities of the environment
            env.P[s][a] is a list of transition tuples (prob,next_state,reward,done)
            env.nS is a number of states in the environment
            env.nA is a number of actions in the environment
        theta:stop evaluation once our value function change is less than theta for all states
        gamma:discount factor
        
    returns:
        a tuple (policy,V) of the optimal policy and the optimal value function    
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        delta = 0
        for state in env.P:
            v_old = copy.deepcopy(V[state])
            v_new = np.zeros(len(env.P[state]))
            for action in env.P[state]:
                for transition_tuple in env.P[state][action]:
                    prob, next_state, reward, done = transition_tuple
                    v_new[action] += prob * (reward + gamma * V[next_state])
            V[state] = np.max(v_new)
            delta = np.maximum(delta, np.abs(V[state] - v_old))
        if delta < theta:
            break

    for state in env.P:
        new_policies = np.zeros(env.nA)
        for action in env.P[state]:
            q_value = 0.0
            for transition_tuple in env.P[state][action]:
                prob, next_state, reward, done = transition_tuple
                q_value += prob * (reward + gamma * V[next_state])
            new_policies[action] = q_value
        policy[state] *= 0.0
        policy[state][np.argmax(new_policies)] = 1

    return policy, V
