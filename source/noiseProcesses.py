# @keras-rl
'''
Script for custom or modified noise processes
'''
from __future__ import division

import numpy as np
         
#makes an instance of a noise process and returns it
#defined by configuration nc
#size is the number of parameters the noise is applied to
#so far just one-dimensional vector (only action noise)
def getNoise(nc, size):
    dictionary = {
        'GWN' : GaussianWhiteNoiseProcess,
        'OU' : OrnsteinUhlenbeckProcess,
        'OUAR' : OUAnnealReset,
        #'AOU' : AlternatingOU
        }
    assert nc['key'] in dictionary, "noise process does not exist"
    
    if nc['key'] == 'GWN':
        mu = nc['mu'] if 'mu' in nc else 0.
        sigma = nc['sigma'] if 'sigma' in nc else 1.
        sigma_min = nc['sigma_min'] if 'sigma_min' in nc else None
        n_steps_annealing = nc['n_steps_annealing'] if 'n_steps_annealing' in nc else 1000
        return GaussianWhiteNoiseProcess(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing, size=size)
    elif nc['key'] == 'OU':
        assert 'theta' in nc
        theta = nc['theta']
        mu = nc['mu'] if 'mu' in nc else 0.
        sigma = nc['sigma'] if 'sigma' in nc else 1.
        sigma_min = nc['sigma_min'] if 'sigma_min' in nc else None
        n_steps_annealing = nc['n_steps_annealing'] if 'n_steps_annealing' in nc else 1000
        dt = nc['dt'] if 'dt' in nc else 1e-2
        #x0 = np.random.normal(mu,sigma,size) if 'x0' in nc else None
        return OrnsteinUhlenbeckProcess(theta=theta, mu=mu, sigma=sigma, sigma_min=sigma_min,
                                        n_steps_annealing=n_steps_annealing, dt=dt, x0=None, size=size)
    elif nc['key'] == 'OUAR':
        assert 'theta' in nc
        theta = nc['theta']
        mu = nc['mu'] if 'mu' in nc else 0.
        sigma = nc['sigma'] if 'sigma' in nc else 1.
        sigma_min = nc['sigma_min'] if 'sigma_min' in nc else None
        n_steps_annealing = nc['n_steps_annealing'] if 'n_steps_annealing' in nc else 1000
        dt = nc['dt'] if 'dt' in nc else 1e-2
        return OUAnnealReset(theta=theta, mu=mu, sigma=sigma, sigma_min=sigma_min,
                             n_steps_annealing=n_steps_annealing, dt=dt, size=size)
        
#### From keras-rl: ####
#the following 4 classes
#https://github.com/keras-rl/keras-rl/blob/1e915aa1943086e3c75c6aaf51b84c6b649c2600/rl/random.py 

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample
       
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
        
#### Own Noise Processes ####
        
#improves the keras-rl OU implementation by making the reset dependent on the standard deviation     
class OUAnnealReset(OrnsteinUhlenbeckProcess):
    def __init__(self,**kwargs):
        super(OUAnnealReset,self).__init__(**kwargs)
    
    def reset_states(self):
        self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size) 


#### Experimentals for fun ####

#Ornstein Uhlenbeck which resets the annealing sigma to the initial value 
class AlternatingOU(OUAnnealReset):
    def __init__(self, n_res, n_steps_annealing, n_begin=0, **kwargs):
        self.n_res = n_res
        self.n_ann = n_steps_annealing
        self.n_begin = n_begin              #step count when to begin with noise
        super(AlternatingOU, self).__init__(n_steps_annealing=n_steps_annealing, **kwargs)
        
    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps % (self.n_ann + self.n_res)) + self.c)
        return sigma
    
    def sample(self):
        if self.n_begin <= 0:
            return super(AlternatingOU, self).sample()
        else:
            self.n_begin -= 1
            return np.random.normal(self.mu, self.sigma_min, self.size)

#OU modification which sets the ongoing output of the noise process to zero
#but does not interrupt the process itself
class PausingOU(OrnsteinUhlenbeckProcess):
    def __init__(self, noiseLength, noisePause, alpha, **kwargs):
        self.alpha = alpha
        self.noiseLength = noiseLength
        self.noisePause = noisePause
        self.np = noisePause + noiseLength
        super(PausingOU, self).__init__(**kwargs)
        
    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        
        if self.np <= self.noisePause:
            x = x * self.alpha
        self.np -= 1
        if self.np <= 0:
            self.np = self.noiseLength + self.noisePause
        
        return x


#testing
if __name__=='__main__':
    from config import noiseConfig as nc
    noise = getNoise(nc[0],5)
    print(noise.theta)
    print(noise.sample())
    noise.reset_states()
    print(noise.x_prev)