'''
An actor/worker which gets a policy, executes it with exploration noise for 
one episode and sends collected memory back
'''

import numpy as np

from config import noiseConfig, getActor, getEnv
from source.noiseProcesses import getNoise

def runExplorer(ID, weightsQueue, expQueue, abort, weightNoiseProb=0.):
    try:
        env = getEnv()
        env.reset()
        nc = noiseConfig[ID % len(noiseConfig)]
        noise = getNoise(nc, env.noutput)
        
        exp = Explorer(ID=ID,env=env,weightNoiseProb=weightNoiseProb,noiseProcess=noise)
        exp.build()
        
        while not abort.value:
            weights = weightsQueue.get()
            exp.model.set_weights(weights)
            results = exp.runOneEpisode(abort)
            expQueue.put(results)
    except KeyboardInterrupt:
        pass
    
class Explorer(object):
    def __init__(self, ID, env, weightNoiseProb, noiseProcess=None):
        self.ID = ID
        self.env = env
        self.wnprob = weightNoiseProb
        self.noise = noiseProcess
        
        self.model = getActor(env)
        
    def build(self):
        self.model.compile(optimizer='sgd', loss='mse')
        
    def runOneEpisode(self, abort):
        episodeMemory = []
        episodeReward = 0.
        episodeInfo = {}
        
        state = self.env.reset()
        if self.noise is not None:
            self.noise.reset_states()
            
        done = False
        actionNoise = np.random.rand() < 1. - self.wnprob
        
        if not actionNoise:
            #TODO: refine parameter noise
            currWeights = self.model.get_weights()
            weights = [w + np.random.normal(scale=1.0, size=np.shape(w)) for w in currWeights]
            self.model.set_weights(weights)
            
        while not (done or abort.value):
            action = self.model.predict_on_batch(np.array([[state]])).flatten()
            
            assert action.shape == self.env.action_space.shape
            if self.noise is not None and actionNoise:
                action += self.noise.sample()
            np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            obs, rew, done, info = self.env.step(action)
            episodeReward += rew
            episodeMemory.append([state,action,rew,done])
            state = obs
            for key in info:
                if key not in episodeInfo:
                    episodeInfo[key] = []
                episodeInfo[key] += [info[key]]
            
        #follow-up experience of terminal state/previous experience before env.reset
        episodeMemory.append([state,action,0.,False])
        return (self.ID, episodeReward, episodeMemory, episodeInfo)