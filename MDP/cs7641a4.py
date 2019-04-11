import numpy as np
import pprint
import sys
import gym
import itertools
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotting

from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from collections import defaultdict
# from lib.envs.cliff_walking import CliffWalkingEnv
# from lib.envs.gridworld import GridworldEnv
# from lib import plotting

def plot_vi_results(x_axis, vi_times, vi_iterations, save_as = False):

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    sns.barplot(x = x_axis, y = vi_iterations)
    plt.title('Number Iterations')
    plt.xlabel('Shape')
    plt.ylabel('Iterations')
    idx=0
    for i in vi_iterations:
        plt.text(idx, i, round(i,2), color='black', ha="center")
        idx+=1

    plt.subplot(1,2,2)
    sns.barplot(x = x_axis, y = vi_times)
    plt.title('Total Runtime')
    plt.xlabel('Shape')
#     plt.set_ylabel('Time (S)')
    plt.ylabel('Time (S)')
    idx = 0
    for i in vi_times:
        plt.text(idx, i, round(i,2), color='black', ha="center")
        idx+=1
 
    if not save_as:
        plt.subplots_adjust(wspace = 0.5, hspace=0.5)
        plt.show()
    else:
        plt.subplots_adjust(wspace = 0.5, hspace=0.5)
        plt.savefig(save_as)

def plot_pi_results(x_axis, pi_times, pi_iterations, pi_eval_time, save_as = False):

    plt.figure(figsize=(10,5))
    plt.subplots_adjust(hspace=1)

    plt.subplot(1,2,1)
    sns.barplot(x = x_axis, y = pi_iterations)
    plt.title('Number Iterations')
    plt.xlabel('Shape')
    plt.ylabel('Iterations')
    idx=0
    for i in pi_iterations:
        plt.text(idx, i, round(i,2), color='black', ha="center")
        idx+=1

    plt.subplot(1,2,2)
    sns.barplot(x = x_axis, y = pi_times)
    plt.title('Total Runtime')
    plt.xlabel('Shape')
    plt.ylabel('Time (S)')
    for i in pi_times:
        plt.text(pi_times.index(i), i, round(i,2), color='black', ha="center")
    
#     plt.subplot(1,3,3)
#     sns.barplot(x = x_axis, y = pi_eval_time)
#     plt.title('Policy Evaluation Time')
#     plt.xlabel('Shape')
#     plt.ylabel('Time (S)')
#     idx=0
#     for i in pi_eval_time:
#         plt.text(idx, i, round(i,2), color='black', ha="center")
#         idx+=1
    if not save_as:
        plt.subplots_adjust(wspace = 0.5, hspace=0.5)
        plt.show()
    else:
        plt.subplots_adjust(wspace = 0.5, hspace=0.5)
        plt.savefig(save_as)

# https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
# matprint.py Pretty print a matrix in Python 3 with numpy
def matprint(mat, fmt="g"):
    if mat[0][0] in ["←", "↓", "→", "↑"]:
        fmt = 's'# fmt='s' for arrows
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        
def print_policy(V, width=4, height=4):
    table = {0: "←", 1: " ↓", 2: "→", 3: "↑"}
    policy = np.resize(V, (width, height))
    
    # transform using the dictionary
    return np.vectorize(table.get)(policy)

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    #initialize iteration
    n_iter = 0
    s_iter = 0
    eval_time = 0
    while True:
        # Evaluate the current policy
        n_iter += 1
        t = time.clock()
        V = policy_eval_fn(policy, env, discount_factor)
        eval_time = eval_time + (time.clock() - t)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
            s_iter += 1
            
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V, n_iter, eval_time

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
#     V = np.zeros(env.nS)
    V = np.random.rand(env.nS)
    n_iter = 0
    
    while True:
        # Stopping condition
        n_iter += 1
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V, n_iter

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
#             print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats

def perform(Q, grid_world_env, verbose = False):
    grid_world_env.reset()
    if verbose:
        print('Starting Position')
        print('--------')
        grid_world_env._render()
    route = []
    rewards = -1
    while rewards < 0:
        _iter = 0
        state = grid_world_env.s
        next_step = np.argmax(Q[state])
        route.append((state,next_step))
        grid_world_env.step(next_step)
        if verbose:
            print('Iteration:{}'.format(_iter))
            print('--------')
            grid_world_env._render()
        temp_rewards = 0
        for value in grid_world_env.P[grid_world_env.s].values():
            temp_rewards += value[0][2]
        _iter += 1
        rewards = temp_rewards
        
    states = [x[0] for x in route]
    actions = [x[1] for x in route]

    grid = np.arange(grid_world.nS).reshape(grid_world.shape)
    it = np.nditer(grid, flags=['multi_index'])
    outfile = sys.stdout
    print('=== AND THE FINAL ROUTE TAKEN IS! ===')
    while not it.finished:
        s = it.iterindex
        y, x = it.multi_index
        if s in states:
            hit = True 
        else:
            hit = False

        if s in states:
            pos = states.index(s)
            action = actions[pos]
            output = direction(action)
        elif s == 0 or s == grid_world.nS - 1:
            output = " T " 
        else:
            output = " o "
            hit = False

        if x == 0:
            output = output.lstrip()
        if x == grid.shape[1] - 1:
            output = output.rstrip()
        outfile.write(output)
        
        if x == grid.shape[1] - 1:
            outfile.write("\n")

        it.iternext()

def direction(value):
    if value == 0:
        direction = " ^ "
    if value == 1:
        direction = " > "
    if value == 2:
        direction = " v "
    if value == 3:
        direction = " < "
    return direction

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)
 