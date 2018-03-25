import os

import numpy as np

from memory import SequentialMemory

#just adds methods to save or load memory
class sm(SequentialMemory):
    def __init__(self, limit, **kwargs):
        super(sm, self).__init__(limit, **kwargs)
        
    def getPath(self, path, load=False):
        p, extension = os.path.splitext(path)
        action_path = p + "_actionMem" + (".npy" if load else "")
        reward_path = p + "_rewardMem" + (".npy" if load else "")
        terminal_path = p + "_termMem" + (".npy" if load else "")
        observation_path = p + "_obsMem" + (".npy" if load else "")
        
        return action_path, reward_path, terminal_path, observation_path
        
    def load(self, path):
        def load_data(rb, path):
            def find_length(v):
                i = len(v) - 1
                while v[i] is None:
                    i -= 1
                return i + 1
                
            rb.data = np.load(path)
            if len(rb.data) > rb.maxlen:
                rb.data = np.array_split(rb.data, [rb.maxlen])[0]
                rb.length = rb.maxlen
            else:
                rb.length = find_length(rb.data)
                rb.data = np.concatenate((rb.data,[None for _ in range(rb.maxlen-len(rb.data))]))
                
        ap, rp, tp, op = self.getPath(path, load = True)
        load_data(self.actions,ap)
        load_data(self.rewards,rp)
        load_data(self.terminals,tp)
        load_data(self.observations,op)        
        
    def save(self, path):
        ap, rp, tp, op = self.getPath(path)
        np.save(ap, self.actions.data)
        np.save(rp, self.rewards.data)
        np.save(tp, self.terminals.data)
        np.save(op, self.observations.data)
        