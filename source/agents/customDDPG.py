# @keras-rl
# https://github.com/Frawak/keras-rl/blob/master/LICENSE

from __future__ import division
import os
import warnings

import numpy as np
import Queue

import keras.backend as K
import keras.optimizers as optimizers
from keras.callbacks import History
from source.callbacks import CallbackList, CollectiveLogger

from source.util import mean_q, clone_model, clone_optimizer, huber_loss, get_soft_target_model_updates, AdditionalUpdatesOptimizer

''' Abstract class for custom DDPG designs
    
    derived from keras-rl DDPGAgent
    https://github.com/keras-rl/keras-rl/blob/3dcd547f8f274f04fe11e813f52ceaed8987c90a/rl/agents/ddpg.py
'''
class CustomDDPG(object):
    EPOCH_PERIOD = 10   #TODO: make variable
    
    #keras-rl: reduced (and modified) __init__ of DDPGAgent and its parent class Agent
    def __init__(self, weightNoiseProb, nb_actions, actor, critic, critic_action_input, 
                 memory, gamma=.99, batch_size=32, delta_range=None, delta_clip=np.inf,
                 custom_model_objects={}, target_model_update=.001):
        self.training = False
        self.step = 0
        
        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]
        
        # Parameters.
        self.nb_actions = nb_actions
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.custom_model_objects = custom_model_objects
        
        self.weightNoiseProb = weightNoiseProb
        
        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory
        
        # State.
        self.compiled = False
    
    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase
    
    #keras-rl: directly copied from DDPGAgent
        #TODO: Overwrite/Change that
        #know exactly what you are doing...this is the trickiest part
    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(critic_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(critic_inputs + [K.learning_phase()],
                                             [self.actor(critic_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                critic_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(critic_inputs, [self.actor(critic_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True
                
    
    #randomly determines if agent uses action or parameter noise
    #not yet used by these agents but by workers
    def actionNoisePermission(self):
        return np.random.rand() < 1. - self.weightNoiseProb        

    #replaces backward from parent class
    #update the actors once
    def train(self):            
        experiences = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size
        
        #group experience data
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)
        state0_batch = np.array(state0_batch)
        state1_batch = np.array(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert action_batch.shape == (self.batch_size, self.nb_actions)
        
        #predict on actions according to experience and predict Q values
        target_actions = self.target_actor.predict_on_batch(state1_batch)
        assert target_actions.shape == (self.batch_size, self.nb_actions)
        if len(self.critic.inputs) >= 3:
            state1_batch_with_action = state1_batch[:]
        else:
            state1_batch_with_action = [state1_batch]
        state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
        target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
        assert target_q_values.shape == (self.batch_size,)
        
        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * target_q_values
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)
        
        #TODO: lr update
        #https://arxiv.org/abs/1711.06922 let their learning rate decay
        
        # Perform a single batch update on the critic network.
        if len(self.critic.inputs) >= 3:
            state0_batch_with_action = state0_batch[:]
        else:
            state0_batch_with_action = [state0_batch]
        state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
        self.critic.train_on_batch(state0_batch_with_action, targets) #TODO: metrics
        
        if len(self.actor.inputs) >= 2:
            inputs = state0_batch[:]
        else:
            inputs = [state0_batch]
        if self.uses_learning_phase:
            inputs += [self.training]
        action_values = self.actor_train_fn(inputs)[0]
        assert action_values.shape == (self.batch_size, self.nb_actions)
    
    #fit function to be called externally
    #to be overwritten
    def fit(self):
        pass
      
    #### The following methods are copied from keras-rl DDPGAgent class ####
    
    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())
    
#################################    
##### Trainable DDPG Agents #####
#################################
    
class CollectiveDDPG(CustomDDPG):
    def __init__(self, **kwargs):
        super(CollectiveDDPG,self).__init__(**kwargs)
            
    def fit(self, nb_trainSteps, wQueues, expQueue, testWeightsQueue, testResultsQueue, abort, callbacks=None):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        self.training = True

        #### setting up callbacks ####
        
        callbacks = [] if not callbacks else callbacks[:]
        myLogger = CollectiveLogger()
        callbacks += [myLogger]
        
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        params = {
            'nb_trainSteps': nb_trainSteps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        callbacks.on_train_begin()

        #### initialize training process ####

        step = 0
        epoch = 0   #for callback purposes
        log_episodeReward = []
        log_info = []
        did_abort = False
        try:
            #give all Explorers and the Tester initial weights
            for weightsQueue in wQueues:
                weightsQueue.put(self.actor.get_weights())
            testWeightsQueue.put(self.actor.get_weights())
            
            #start training process
            while step < nb_trainSteps:
                
                #exchange with one Explorer 
                try:
                    explorer, episodeReward, episodeMemory, episodeInfo = expQueue.get_nowait()
                    #give done Explorer next weights
                    wQueues[explorer].put(self.actor.get_weights())
                    #save episodeMemory from Explorer to own memory or training
                    numSteps = len(episodeMemory) 
                    for i in range(numSteps):
                        obs, act, rew, term = episodeMemory[i]
                        self.memory.append(obs,act,rew,term)
                    
                    epoch += 1
                    log_episodeReward.append(episodeReward)
                    log_info.append(episodeInfo)
                    if epoch % self.EPOCH_PERIOD == 0 and epoch > 0:
                        epoch_logs = {
                                'step' : step,
                                'reward_mean' : np.mean(log_episodeReward),
                                'reward_min' : np.min(log_episodeReward),
                                'reward_max' : np.max(log_episodeReward),
                                'environmentInfo' : log_info
                                }
                        callbacks.on_episode_end(epoch/self.EPOCH_PERIOD, epoch_logs)
                        log_episodeReward = []
                        log_info = []
                except Queue.Empty:
                    pass
                  
                #TODO: maybe other criteria for new training step
                if self.memory.nb_entries > self.batch_size:                
                    #callbacks.on_step_begin(step)   
                    
                    self.train()
                    
                    #TODO: training metrics to callback
                    #callbacks.on_step_end(step, step_logs)
                    step += 1
                    
                #exchange with tester
                try:
                    testInfo, testReward = testResultsQueue.get_nowait()
                    testWeightsQueue.put(self.actor.get_weights())
                    test_logs = {
                            'info' : testInfo,
                            'reward' : testReward
                            }
                    callbacks.on_test_end(test_logs)
                except Queue.Empty:
                    pass
                    
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        
        #### post-training ####
        
        abort.value = 1
        callbacks.on_train_end(logs={'did_abort': did_abort})

        return history, myLogger
     
