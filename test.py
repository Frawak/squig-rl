'''
Script to test one policy
'''
import argparse

parser = argparse.ArgumentParser(description='Test neural net motor controller')
parser.add_argument('--episodes', dest='episodes', action='store', default=1, type=int) #number of episode to be tested
parser.add_argument('--dir', dest='dir', action='store', default="../temp/")
parser.add_argument('--model', dest='model', action='store', default="example.h5f")     #filename with weights to be tested
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
tester.loadWeights(args.dir + args.model)
tester.test(env,args.episodes)