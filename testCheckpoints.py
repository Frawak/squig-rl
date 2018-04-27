'''
Script to test the checkpoints of one training session
'''
import argparse
import os

parser = argparse.ArgumentParser(description='Test checkpoints of neural net motor controller')
parser.add_argument('--dir', dest='dir', action='store', default="../temp/")
parser.add_argument('--model', dest='model', action='store', default="example.h5f")         #general model name
parser.add_argument('--start', dest='start', action='store', default=1, type=int)           #First episode-number - args.interval
parser.add_argument('--interval', dest='interval', action='store', default=100, type=int)   #checkpoint interval
parser.add_argument('--obs', dest='obs', action='store_true', default=False)
parser.add_argument('--vis', dest='visualize', action='store_true', default=False)
args = parser.parse_args()

from source.agents.testActor import TestActor
from config import getActor, getEnv, ENV_TAG

env = getEnv(visualize=args.visualize)
env.reset()

if args.visualize:
    if ENV_TAG=='OSIM':
        vis = env.osim_model.model.updVisualizer().updSimbodyVisualizer()
        vis.setBackgroundType(vis.GroundAndSky)        

actor = getActor(env)
print(actor.summary())

tester = TestActor(actor)
epoch = args.start
log = []
while True:
    try:
        epoch += args.interval
        print("epoch {:04d}".format(epoch))
        filename, extension = os.path.splitext(args.dir + args.model)
        path = filename + '_{:04d}'.format(epoch) +extension
        tester.loadWeights(path)
        obs_log = tester.test(env,1)    #TODO: multiple episode test per checkpoint
        if args.obs:
            log.append(obs_log)
    except OSError:
        break
    
try:
    print("Final")
    tester.loadWeights(args.dir + args.model)
    obs_log = tester.test(env,1)    #TODO: ...
    if args.obs:
        log.append(obs_log)
except OSError:
    pass
    
if args.obs:
    import numpy as np
    for obs_log in log:
        print(np.max(obs_log, axis=0))
        print(np.min(obs_log, axis=0))
    
if ENV_TAG=='GYM' and args.visualize:
    env.close()
    