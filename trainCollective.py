'''
Training script for CollectiveDDPG
'''

import argparse
import os

#Command line parameters
parser = argparse.ArgumentParser(description='Train neural net motor controller')
#Number of episodes for training
parser.add_argument('--steps', dest='steps', action='store', default=100000000, type=int)       #training steps
#Number of explorers
parser.add_argument('--nexp', dest='numExplorers', action='store', default=2, type=int)         #number of Explorer threads
#Saved stuff
parser.add_argument('--dir', dest='dir', action='store', default="../temp/")                    #directory to store all saved files
parser.add_argument('--model', dest='model', action='store', default="example.h5f")             #file name to store final weights in
parser.add_argument('--weights', dest='weights', action='store', default=None)                  #existing weights to be loaded
parser.add_argument('--memory', dest='memory', action='store', default=None)                    #existing memory to be loaded

args = parser.parse_args()

filename, extension = os.path.splitext(args.model)

from config import getActorCritic, getEnv, gamma, tau, batch_size, wnprob, memory, optimizer

#Check if directory exists and if not, create it
if not os.path.exists(args.dir):
    os.mkdir(args.dir)

#### Setting up multiprocessing primitives ####

from multiprocessing import Process, Queue, Value
abort = Value('i', 0)
wQueues = []
expQueue = Queue()
testWeights = Queue()
testResults = Queue()

#### Start child processe ####

from source.agents.explorer import runExplorer
for i in range(args.numExplorers):
    weightsQueue = Queue()
    p = Process(target=runExplorer, args=(i, weightsQueue, expQueue, abort, wnprob,))
    p.daemon = True
    p.start()
    wQueues.append(weightsQueue)
    
from source.agents.tester import runTester
p = Process(target=runTester, args=(testWeights,testResults,abort,))
p.daemon = True
p.start()

#### Initiate main process agent for training ####
    
from source.customKerasLayers import LayerNormDense
from source.agents.customDDPG import CollectiveDDPG
env = getEnv()
env.reset()
numActions = env.action_space.shape[0]
memory_ = memory
optimizer_ = optimizer

actor, critic, metrics_, action_input = getActorCritic(env)

print(actor.summary())
print(critic.summary())

agent = CollectiveDDPG(weightNoiseProb=wnprob, nb_actions=numActions, 
                       actor=actor, critic=critic, critic_action_input=action_input,
                       memory=memory_, gamma=gamma, batch_size=batch_size,
                       delta_clip=1., target_model_update=tau,
                       custom_model_objects={"LayerNormDense" : LayerNormDense})

agent.compile(optimizer_, metrics=metrics_)

#### Additional callbacks ####

from keras.callbacks import ModelCheckpoint
mc = ModelCheckpoint(args.dir + filename + '_{epoch:04d}' + extension, save_weights_only=True, period=100)  #TODO: make period into argument
callbacks = [mc]

#### Load previous weights and memory for training ####

if args.weights is not None:
    agent.load_weights(args.dir + args.weights)
    
if args.memory is not None:
    agent.memory.load(args.memory)
    
#### Training ####
    
history, myLogger = agent.fit(args.steps, wQueues, expQueue, testWeights, testResults, abort, callbacks)
    
#### Post-Training ####

#Save weights after training
print("Saving Weights")
agent.save_weights(args.dir + args.model, overwrite=True)

#Save logs
print("Saving Logs")
import pickle
pickle.dump(myLogger.history, open(args.dir + filename + "_logs.p", "wb"))
pickle.dump(history.history, open(args.dir + filename + "_history.p", "wb"))

#Save memory
try:
    print("Saving memory")
    print("This will take a while. To interrupt press Ctrl+C once!")
    agent.memory.save(args.dir + filename)
except:
    print("Memory saving interrupted or failed")
    pass

#### Plotting ####
from plotLogs import plotCustom
print("Plotting")
plotCustom(destinyFile=args.dir + filename, logs=myLogger.history, 
           abscissaKey='duration', keys=['reward'],
           xLabel='Duration', yLabel='Distance')

print("FINISHED")
