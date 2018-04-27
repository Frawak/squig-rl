import numpy as np
'''
Agent with just the actor network and feed-forwarding information 
which executes actions without noise.
'''  
class TestActor(object):
    def __init__(self, model):
        self.model = model
        self.model.compile(optimizer='sgd', loss='mse')
        
    def test(self, env, numEpisodes=1):
        print("Test for "+str(numEpisodes)+" episodes")
        obs_log = []
        
        for i in range(numEpisodes):
            state = env.reset()
            done = False
            step = 0
            episodeReward = 0.
            
            while not done:
                action = self.model.predict_on_batch(np.array([[state]])).flatten()
                assert action.shape == env.action_space.shape
                np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, rew, done, info = env.step(action)
                
                episodeReward += rew
                state = obs
                step += 1
                
                obs_log.append(obs)
                
            print("episode "+str(i+1)+", steps: "+str(step)+", reward: "+str(episodeReward))
            # TODO: env info generalize
            #+", distance: "+str(env.getDistanceTravelled()))
            
        return obs_log
            
    def loadWeights(self, filepath):
        import os
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        self.model.load_weights(actor_filepath)
