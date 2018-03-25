'''
Script to plot a noise process
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from source.noiseProcesses import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess, PausingOU, AlternatingOU

parser = argparse.ArgumentParser(description='Plot noise over time')
#Annealing Noise Process Parameters
parser.add_argument('--theta', dest='theta', action='store', default=.1, type=float)
parser.add_argument('--mu', dest='mu', action='store', default=0., type=float)
parser.add_argument('--sigma', dest='sigma', action='store', default=.2, type=float)
parser.add_argument('--dt', dest='dt', action='store', default=1e-2, type=float)
parser.add_argument('--sigma_min', dest='sigma_min', action='store', default=.05, type=float)
parser.add_argument('--anneal', dest='anneal', action='store', default=1e6, type=int)
#Parameters for some fun experimentals
parser.add_argument('--alpha', dest='alpha', action='store', default=1e-3, type=float)
parser.add_argument('--nl', dest='nl', action='store', default=500, type=int)
parser.add_argument('--np', dest='np', action='store', default=500, type=int)

parser.add_argument('--n_res', dest='n_res', action='store', default=200000, type=int)
parser.add_argument('--n_begin', dest='n_begin', action='store', default=10000, type=int)
#Plot scale and dimension of the noise output
parser.add_argument('--scale', dest='scale', action='store', default=1.5, type=float)   #plot ordinate scale
parser.add_argument('--size', dest='size', action='store', default=1, type=int)         #vector size
parser.add_argument('--range', dest='range', action='store', default=1.3e6, type=int)   #abscissa scale
args = parser.parse_args()

noise1 = OrnsteinUhlenbeckProcess(theta=args.theta, mu=args.mu, sigma=args.sigma, size=args.size, 
                                  #x0=np.random.normal(args.mu,args.sigma,args.size),
                                  #x0=np.random.normal(args.mu,1.0,args.size),
                                  dt=args.dt, n_steps_annealing=args.anneal, sigma_min = args.sigma_min)

noise2 = GaussianWhiteNoiseProcess(mu=args.mu, sigma=args.sigma, size=args.size, n_steps_annealing=args.anneal,
                                   sigma_min = args.sigma_min)

noise3 = PausingOU(noiseLength=args.nl, noisePause=args.np, alpha=args.alpha,
                      theta=args.theta, mu=args.mu, sigma=args.sigma, size=args.size,
                      x0=np.random.normal(args.mu,args.sigma,args.size),
                      dt=args.dt, n_steps_annealing=args.anneal, sigma_min = args.sigma_min)

noise4 = AlternatingOU(n_begin=args.n_begin, n_res=args.n_res, theta=args.theta, mu=args.mu, sigma=args.sigma, 
                       size=args.size,  x0=np.random.normal(args.mu,args.sigma,args.size),
                       dt=args.dt, n_steps_annealing=args.anneal, sigma_min = args.sigma_min)

noise = noise4      #TODO: select noise process here
f = [noise.sample()] 
x = [0]
for i in range(int(args.range)):
    if i % 1000 == 0:           #TODO: make reset interval into argument
        noise.reset_states()
    f += [noise.sample()]
    x += [i+1]


plt.plot(x,f)
plt.grid()
plt.axhline(0, color='black')
plt.ylim(ymin=-args.scale, ymax=args.scale)
plt.show()