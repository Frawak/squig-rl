

#a wrapper in order to visualize gym environments with their built-in render method
class gymVisualizer(object):
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.env.render(mode='human')
        return obs, rew, done, info
    
    def __getattr__(self, name):
        return self.env.__getattribute__(name)