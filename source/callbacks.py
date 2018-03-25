# @keras-rl
# https://github.com/Frawak/keras-rl/blob/master/LICENSE

'''
Callbacks for logging purposes
    are derived from keras-rl callbacks
    look also into keras callbacks: https://keras.io/callbacks/
    
    see e.g. customDDPG.py for their use
'''

import timeit
import numpy as np

from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList

### Callback & CallbackList are taken from keras-rl and modified ###

class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        pass

    def on_episode_end(self, episode, logs={}):
        pass

    def on_step_begin(self, step, logs={}):
        pass

    def on_step_end(self, step, logs={}):
        pass

    def on_action_begin(self, action, logs={}):
        pass

    def on_action_end(self, action, logs={}):
        pass

    def on_test_begin(self, action, logs={}):
        pass

    def on_test_end(self, action, logs={}):
        pass

class CallbackList(KerasCallbackList):
    def _set_env(self, env):
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)

    def on_episode_begin(self, episode, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            else:
                callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            else:
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)
                
    def on_test_begin(self, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_test_begin', None)):
                callback.on_test_begin(logs=logs)
                
    def on_test_end(self, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_test_end', None)):
                callback.on_test_end(logs=logs)
                
''' #needed for other environemnts?
class Visualizer(Callback):
    def on_action_end(self, action, logs):
        self.env.render(mode='human')
'''
                
### Custom Callbacks ###
                
class CollectiveLogger(Callback):
    def __init__(self):
        pass
    
    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        print('Training for {} steps ...'.format(self.params['nb_trainSteps']))
        self.history = {}
        
    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.train_start
        template = 'Epoch {epoch:d}, Step {step:d}, duration: {duration:.3f}s, reward: {rew_mean:.3f} [{rew_min:.3f}, {rew_max:.3f}]'
        variables = {
                'epoch' : int(episode),
                'step' : logs['step'],
                'duration' : duration,
                'rew_mean': logs['reward_mean'],
                'rew_max': logs['reward_max'],
                'rew_min': logs['reward_min']
                }
        print(template.format(**variables))
        
        additionalInfo = logs['environmentInfo'] #array (explorer) of dictionaries (episodeInfo) of arrays (stepInfo)
        for key in additionalInfo[0]:
            lastEntries = []
            for dictionary in additionalInfo:
                episodeInfo = dictionary[key]
                lastEntries.append(episodeInfo[len(episodeInfo) - 1])
            template = key + ': {mean:.3f} [{min:.3f}, {max:.3f}]'
            variables = {
                    'mean' : np.mean(lastEntries),
                    'min' : np.min(lastEntries),
                    'max' : np.max(lastEntries)
                    }
            print(template.format(**variables))
                
        #TODO: more than just the last entries of the episodeInfos
        #differ between episode relevant and step relevant information
        #TODO: be cautious about other info than float values
        
    def on_test_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        duration /= 3600.
        self.history.setdefault('duration', []).append(duration)
        self.history.setdefault('reward', []).append(logs['reward'])
        template = 'Test at {duration:.3f}h, reward: {rew:.3f}'
        variables = {
                'duration': duration,
                'rew' : logs['reward']
                }
        info = logs['info']
        for key in info:
            episodeInfo = info[key]
            lastEntry = episodeInfo[len(episodeInfo) - 1]
            variables[key] = lastEntry
            template += ', ' + key + ': {' + key + ':.3f}'
            self.history.setdefault(key, []).append(lastEntry)
            
        print(template.format(**variables))
        
    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))
