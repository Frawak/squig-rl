'''
Script to test the checkpoints of one training session
'''
import argparse
import os

parser = argparse.ArgumentParser(description='Test checkpoints of neural net motor controller')
parser.add_argument('--dir', dest='dir', action='store', default="../temp/")
parser.add_argument('--model', dest='model', action='store', default="example.h5f")         #general model name
parser.add_argument('--start', dest='start', action='store', default=0, type=int)           #First episode-number - args.interval
parser.add_argument('--interval', dest='interval', action='store', default=100, type=int)   #checkpoint interval
parser.add_argument('--obs', dest='obs', action='store_true', default=False)
parser.add_argument('--not_visualize', dest='visualize', action='store_false', default=True)
args = parser.parse_args()

from source.agents.testActor import TestActor
from config import getActor, getEnv

env = getEnv(visualize=args.visualize)
env.reset()

#currently osim specific
try:
    vis = env.osim_model.model.updVisualizer().updSimbodyVisualizer()
    vis.setBackgroundType(vis.GroundAndSky)
except:
    pass

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
        obs_log = tester.test(env,1)
        if args.obs:
            log.append(obs_log)
    except:
        break
    
if args.obs:
    import numpy as np
    for obs_log in log:
        print(np.max(obs_log, axis=0))
        print(np.min(obs_log, axis=0))