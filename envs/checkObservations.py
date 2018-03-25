'''
Script to print observation values of osim Running Environment
'''

import numpy as np
from envs.diffEnv import diffEnv
from osim.env import RunEnv

env = RunEnv(max_obstacles = 10)
obs = env.reset(difficulty = 2, seed=47)

def print_obs(obs):
    #iterate = range(len(obs))
    iterate = [29,31,33,35]
    for i in iterate:
        print (str(i) + ": " + str(obs[i]))

try:
    print_obs(obs)
    for t in range(200):
        action= np.zeros(18)
        obs, _, done, _ = env.step(action)
        print("")
        print_obs(obs)
        if done: break
    
    while True:
        pass
except KeyboardInterrupt:
    pass