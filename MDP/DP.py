
import numpy as np
import pandas as pd
from envs import GridworldEnv, WindyFrozenLake
from cs7641a4 import policy_eval, policy_improvement, value_iteration, matprint, print_policy, plot_pi_results, plot_vi_results
import time
import seaborn as sns
import matplotlib.pyplot as plt

gw64 = GridworldEnv(shape = [8,8])
fl256 = WindyFrozenLake(map_name="16x16")
fl64 = WindyFrozenLake(map_name="8x8")
envs = [(gw64, gw64.shape), (fl64,[8,8]), (fl256, [16,16])]


## discount factor of 0.9
discount_factor = 0.9

## policy iteration
pi_times = []
pi_iterations = []
pi_eval_time = []
pi_1_policies = []
for i in envs:
    
    env = i[0]
    shape = i[1]
    
    
    t = time.clock()
    pi_policy, v, iterations, eval_time = policy_improvement(env, discount_factor = discount_factor)
    pi_1_policies.append(pi_policy)
    pi_times.append(time.clock() - t)
    pi_iterations.append(iterations)
    pi_eval_time.append(eval_time)
    
    
    print("Reshaped Grid Policy (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP):")
    print(np.reshape(np.argmax(pi_policy, axis=1), shape))
    pi = np.argmax(pi_policy, axis=1)
    print("")
    
plot_pi_results(x_axis = ["GW 8x8", "FL 8x8","FL 16X16"], 
                pi_iterations = pi_iterations, 
                pi_times = pi_times, 
                pi_eval_time = pi_eval_time,
                save_as = 'PI_1.png')

pi_policy, v, iterations, eval_time = policy_improvement(gw64)
pi = np.argmax(pi_policy, axis=1)
matprint(print_policy(pi, width=8, height=8))


## value iteration
vi_times = []
vi_iterations = []
vi_1_policies = []
for i in envs:

    env = i[0]
    shape = i[1]
    
    t = time.clock()
    vi_policy, v, iterations = value_iteration(env)
    vi_1_policies.append(vi_policy)
    vi_times.append(time.clock() - t)
    vi_iterations.append(iterations)

    print("Reshaped Grid Policy (0=left, 1=down, 2=right, 3=up):")
    print(np.reshape(np.argmax(vi_policy, axis=1), shape))
    print("")
    
plot_vi_results(x_axis = ["GW 8x8", "FL 8x8","FL 16X16"], 
                vi_iterations = vi_iterations, 
                vi_times = vi_times,
                save_as = "VI_1.png")

vi_policy, v, iterations = value_iteration(gw64)
pi = np.argmax(vi_policy, axis=1)
matprint(print_policy(pi, width=8, height=8))

## discount factor of 0.5

discount_factor = 0.5

pi_times = []
pi_iterations = []
pi_eval_time = []
pi_2_policies = []
for i in envs:
    
    env = i[0]
    shape = i[1]
    
    
    t = time.clock()
    pi_policy, v, iterations, eval_time = policy_improvement(env, discount_factor = discount_factor)
    pi_2_policies.append(pi_policy)
    pi_times.append(time.clock() - t)
    pi_iterations.append(iterations)
    pi_eval_time.append(eval_time)
    
    
    print("Reshaped Grid Policy (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP):")
    print(np.reshape(np.argmax(pi_policy, axis=1), shape))
    pi = np.argmax(pi_policy, axis=1)
    print("")
    
plot_pi_results(x_axis = ["GW 8x8", "FL 8x8","FL 16X16"], 
                pi_iterations = pi_iterations, 
                pi_times = pi_times, 
                pi_eval_time = pi_eval_time,
                save_as = 'PI_2.png')

pi_policy, v, iterations, eval_time = policy_improvement(gw64)
pi = np.argmax(pi_policy, axis=1)
matprint(print_policy(pi, width=8, height=8))

vi_times = []
vi_iterations = []
vi_2_policies = []
for i in envs:

    env = i[0]
    shape = i[1]
    
    t = time.clock()
    vi_policy, v, iterations = value_iteration(env, discount_factor = discount_factor)
    vi_2_policies.append(vi_policy)
    vi_times.append(time.clock() - t)
    vi_iterations.append(iterations)

    print("Reshaped Grid Policy (0=left, 1=down, 2=right, 3=up):")
    print(np.reshape(np.argmax(vi_policy, axis=1), shape))
    print("")
    
plot_vi_results(x_axis = ["GW 8x8", "FL 8x8","FL 16X16"], 
                vi_iterations = vi_iterations, 
                vi_times = vi_times,
                save_as = 'VI_2.png')

vi_policy, v, iterations = value_iteration(gw64)
pi = np.argmax(vi_policy, axis=1)
matprint(print_policy(pi, width=8, height=8))
