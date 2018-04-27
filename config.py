#### Environment ####
#Import and configure the used environment here
#Change the import and parameters in getEnv in order to do so

ENV_TAG='ROBOSCHOOL'

if ENV_TAG=='OSIM':
    from envs.diffEnv import diffEnv as env
if ENV_TAG=='GYM' or 'ROBOSCHOOL':
    import gym
    #from gym import wrappers
    from envs.gymWrapper import gymVisualizer as gv
if ENV_TAG=='ROBOSCHOOL':
    import roboschool

def getEnv(visualize=False):
    if ENV_TAG=='OSIM':
        e = env(difficulty=0, visualize=visualize, rewardMode=15, rewardScale=1.,
                action_repetition=3, pel_min=0.6)
    elif ENV_TAG=='GYM' or 'ROBOSCHOOL':
        gym.undo_logger_setup()
        e = gym.make('RoboschoolInvertedPendulum-v1')
        #e.seed(0)
        if visualize:
            e = gv(e)        
    else:
        e = None
        raise RuntimeError('No valid environment selected!')
    
    return e

#### Noise ####
#for multiple Explorers with individual noise processes
noiseConfig = [
        {'key' : 'OUAR', 'theta' : 0.1, 'mu' : 0., 'sigma' : 0.2, 'sigma_min' : 0.05, 'dt' : 1e-2, 'n_steps_annealing' : 1e6},
        {'key' : 'OUAR', 'theta' : 0.15, 'mu' : 0., 'sigma' : 0.15, 'sigma_min' : 0.025, 'dt' : 1e-2, 'n_steps_annealing' : 1e8},
        {'key' : 'GWN', 'sigma' : 0.2, 'sigma_min' : 0.05, 'n_steps_annealing' : 1e6},
        {'key' : 'GWN', 'sigma' : 0.15},
        ]

#### Learning Hyper Parameters ####
gamma = 0.96        #Discount Factor
tau = 0.001         #Target Update Factor
batch_size = 128
wnprob = 0.3        #Weight-/Parameter Noise Probability

#### Memory/Replay Buffer ####
from source.memorySaver import sm
memory = sm(limit=1000000, window_length=1)

#### Network ####

networkID = 1

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, LeakyReLU
from source.customKerasLayers import LayerNormDense

optimizer = [Adam(lr=3e-4, clipnorm=1.),Adam(lr=3e-4)]

metrics = ['mae']

def getActor(env):
    numActions = env.action_space.shape[0]
    
    if networkID == 1:
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(LayerNormDense(128))
        actor.add(Activation('selu'))
        actor.add(LayerNormDense(128))
        actor.add(Activation('selu'))
        actor.add(LayerNormDense(64))
        actor.add(Activation('selu'))
        actor.add(LayerNormDense(64))
        actor.add(Activation('selu'))
        actor.add(Dense(numActions))
        actor.add(Activation('sigmoid'))
    elif networkID == 2:
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(LayerNormDense(64))
        actor.add(Activation('elu'))
        actor.add(LayerNormDense(64))
        actor.add(Activation('elu'))
        actor.add(Dense(numActions))
        actor.add(Activation('sigmoid'))
    elif networkID == 3:
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(LayerNormDense(256))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(LayerNormDense(128))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(LayerNormDense(128))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(Dense(numActions))
        actor.add(Activation('sigmoid'))
    elif networkID == 4:
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(LayerNormDense(16))
        actor.add(Activation('selu'))
        actor.add(LayerNormDense(16))
        actor.add(Activation('selu'))
        actor.add(Dense(numActions))
        actor.add(Activation('linear'))
    else:
        raise Exception('NO EXISTING DESIGN CHOSEN')
        return None
    
    return actor

def getCritic(env):
    numActions = env.action_space.shape[0]
    action_input = Input(shape=(numActions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    
    if networkID == 1:
        x = LayerNormDense(128)(flattened_observation)
        x = Activation('selu')(x)
        x = concatenate([x, action_input])
        x = LayerNormDense(128)(x)
        x = Activation('selu')(x)
        x = LayerNormDense(64)(x)
        x = Activation('selu')(x)
        x = LayerNormDense(64)(x)
        x = Activation('selu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
    elif networkID == 2:
        x = concatenate([flattened_observation, action_input])
        x = LayerNormDense(64)(x)
        x = Activation('tanh')(x)
        x = LayerNormDense(32)(x)
        x = Activation('tanh')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
    elif networkID == 3:
        x = LayerNormDense(256)(flattened_observation)
        x = LeakyReLU(alpha=0.2)(x)
        x = concatenate([x, action_input])
        x = LayerNormDense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = LayerNormDense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
    elif networkID == 4:
        x = LayerNormDense(16)(flattened_observation)
        x = Activation('selu')(x)
        x = concatenate([x, action_input])
        x = LayerNormDense(16)(x)
        x = Activation('selu')(x)
        x = LayerNormDense(16)(x)
        x = Activation('selu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
    else:
        raise Exception('NO EXISTING DESIGN CHOSEN')
        return None
    
    return critic, action_input

def getActorCritic(env):
    critic, action_input = getCritic(env)
    return getActor(env), critic, metrics, action_input
