'''
Script to test one policy
'''
import argparse

parser = argparse.ArgumentParser(description='Test neural net motor controller')
parser.add_argument('--episodes', dest='episodes', action='store', default=1, type=int) #number of episode to be tested
parser.add_argument('--dir', dest='dir', action='store', default="../temp/")
parser.add_argument('--model', dest='model', action='store', default="example.h5f")     #weights
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
tester.loadWeights(args.dir + args.model)
tester.test(env,args.episodes)

if ENV_TAG=='GYM' and args.visualize:
    env.close()