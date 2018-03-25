'''
Like explorer.py just without noise and other exchanges with main process
'''

import numpy as np

from config import getActor, getEnv

def runTester(testWeightsQueue, resultQueue, abort):
    try:
        env = getEnv()
        env.reset()
        
        tester = Tester(env)
        tester.build()
        
        while not abort.value:
            weights = testWeightsQueue.get()
            tester.model.set_weights(weights)
            log = tester.runOneEpisode(abort)
            resultQueue.put(log)
    except KeyboardInterrupt:
        pass

class Tester(object):
    def __init__(self, env):
        self.env = env
        
        self.model = getActor(env)
        
    def build(self):
        self.model.compile(optimizer='sgd', loss='mse')
        
    def runOneEpisode(self, abort):
        episodeReward = 0.
        episodeInfo = {}
        
        state = self.env.reset()
            
        done = False
            
        while not (done or abort.value):
            action = self.model.predict_on_batch(np.array([[state]])).flatten()
            assert action.shape == self.env.action_space.shape
            np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            obs, rew, done, info = self.env.step(action)
            episodeReward += rew
            state = obs
            for key in info:
                if key not in episodeInfo:
                    episodeInfo[key] = []
                episodeInfo[key] += [info[key]]
            
        return (episodeInfo, episodeReward)