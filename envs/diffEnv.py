import opensim
import math
import numpy as np

from osim.env import RunEnv, flatten

import source.myHelper as h

'''
Overwrite from the challenge environment
'''

class diffEnv(RunEnv):
    PELVIS_Y_Initial = 0.91
    PELVIS_Y_DONE = 0.65
    
    #Pelvis redundancy at 24,25
    STATE_JOINTS = 6
    STATE_KNEE_R = 7
    STATE_KNEE_L = 10
    STATE_CENTER_MASS_X = 18
    STATE_HEAD_X = 22
    STATE_HEAD_Y = 23
    STATE_TOES_L_Y = 29
    STATE_PSOAS_L = 36
    STATE_OBSTACLE = 38
    
    STATE_V_PELVIS = 3
    STATE_V_JOINTS = 12
    
    N_BODIES = 7
    
    stepsize = 0.01
    
    #dirty
    ninput = 41
    
    def __init__(self, difficulty=2, rewardMode=0, rewardScale=1., action_repetition=1, 
                 pel_min=0.65, redundant=False, vel_add=False, rel_vel_add=True, 
                 acc_add=False, **kwargs):
        self.difficulty = difficulty                    #obstacles and psoas strength
        self.rewardMode = rewardMode                    #see compute_reward
        self.rewardScale = rewardScale                  #factor multiplied to the reward function
        self.action_repetition = action_repetition      #frame skips
        self.pel_min = pel_min                          #y-coordinate threshold of the pelvis under which the episode will be aborted
        self.redundant_obs = redundant                  #either or not the redundancy in the obs vector is removed
        self.vel_add = vel_add                          #either or not world-centered velocity is added to obs
        self.rel_vel_add = rel_vel_add                  #either or not pelivs-centered velocity is added to obs
        self.acc_add = acc_add                          #either or not world-centered acceleration is added to obs
        
        #adjust the size of the final observation vector (input to actor and/or critic)
        self.ninput += 0 + (14 if vel_add else 0) + (14 if rel_vel_add else 0) + (11 if acc_add else 0)
        #adjust indices of > 23 in obs vector if redundancy is removed
        if not redundant:
            self.ninput -= 2 * (1 + (1 if vel_add else 0) + (1 if rel_vel_add else 0))
            self.STATE_OBSTACLE -= 2
            self.STATE_PSOAS_L -= 2
            self.STATE_TOES_L_Y -= 2
            self.N_BODIES -= 1
        
        super(diffEnv, self).__init__(**kwargs)
        
    #overwrite of episode reset
    def reset(self, **kwargs):
        self.pelvis_x = 0.      #unprocessed world x-coordinate of pelvis
        state =  super(diffEnv, self).reset(difficulty=self.difficulty, **kwargs)   #remain at the initial difficulty
        self.hy_initial = self.current_state[self.STATE_HEAD_Y] #save initial head height for reward/penalty calculation
        
        return state
    
    #overwrite: not a hard coded value but instance-modifiable
    def is_pelvis_too_low(self):
        return (self.current_state[self.STATE_PELVIS_Y] < self.pel_min)
    
    #overwrite: apply frame skip here instead at agent's fit and test function
    #caution with the underscore methods: https://github.com/stanfordnmbl/osim-rl/issues/92
    def _step(self, action):
        reward = 0
        for _ in range(self.action_repetition):
            s,r,t,_ = super(diffEnv, self)._step(action)
            reward += r
            if t:
                break
        dist = self.getDistanceTravelled()
        return s, reward*self.rewardScale, t, {'distance' : dist}
    
    ###############################
    #### Observation overwrite ####
    ###############################
    #make body coordinates relative to pelvis
    #observation expansion inspired by https://github.com/ctmakro/stanford-osrl/blob/master/observation_processor.py
    #take also a look at: https://github.com/AdamStelmaszczyk/learning2run/blob/master/baselines/baselines/pposgd/pposgd_simple.py , https://github.com/fgvbrt/nips_rl/blob/master/state.py
    #ctmakro wrote a wrapper, this here is an overwrite/child class
    #here not (yet) included: expansion of observation by obstacles and foot touch indicators, observation buffer
    #additionally here: removal of observation redundancy for dimensionality reduction
    
    #similar to the get_observation of the parent class
    def get_unredundant_observation(self):
        bodies = ['head', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']    #this is different

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

        jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
        joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]
        joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in range(6)]

        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]  
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for body in bodies]

        muscles = [ self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R] ]
    
        # see the next obstacle
        obstacle = self.next_obstacle()

        self.current_state = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(flatten(body_transforms)) + muscles + obstacle
        return self.current_state
    
    #overwrite of the general observation return
    def get_observation(self):
        #get the raw values from the environment
        if self.redundant_obs:
            obs = super(diffEnv,self).get_observation()
        else:
            obs = self.get_unredundant_observation()
        
        #save initial x-coordinate to record the travelled distance in the episode
        if self.istep == 0: 
            self.initital_pelvis_x = self.current_state[self.STATE_PELVIS_X]
            
        #save previous x-coordinate for reward calculation
        self.prev_pelvis_x = self.pelvis_x
        self.pelvis_x = self.current_state[self.STATE_PELVIS_X]
            
        #save last state (from last frame) for change calculations
        last = self.current_state
        if self.istep > 0:
            last = self.last_state
        
        #if we have no obstacles, set the obs values to zero
        if self.difficulty == 0:
            obs[self.STATE_OBSTACLE]=0
            obs[self.STATE_OBSTACLE+1]=0
            obs[self.STATE_OBSTACLE+2]=0
    
        #make body parts coordinates pelvis-centered
        def pelvis_relative():
            px = obs[self.STATE_PELVIS_X]
            py = obs[self.STATE_PELVIS_Y]
            
            #Pelvis centered observation makes Pelvis coordinates 0
            #Y coordinate unchanged because it's important for falling and bound anyway
                #has to be adjusted in future work if environment is not a plane/axis
            obs[self.STATE_PELVIS_X] = .0
            
            #Bodyparts
            j = self.STATE_HEAD_X
            for i in range(self.N_BODIES):
                obs[j+i*2+0] -= px
                obs[j+i*2+1] -= py
                
            #Center of Mass
            #Position
            obs[self.STATE_CENTER_MASS_X] -= px
            obs[self.STATE_CENTER_MASS_X+1] -= py
            #Velocity
            obs[self.STATE_CENTER_MASS_X+2] -= obs[self.STATE_PELVIS_X+3]
            obs[self.STATE_CENTER_MASS_X+3] -= obs[self.STATE_PELVIS_Y+3]
            
        #add velocity to the obs vector
        def add_bodyPart_vel():            
            add = []
            
            pvel = [(self.current_state[i]-last[i])/self.stepsize for i in range(1,3)]
            bvel = [(self.current_state[i]-last[i])/self.stepsize for i in range(self.STATE_HEAD_X,self.STATE_PSOAS_L)]
            
            #add velocities of other body parts
            #velocity of pelvis already in 
            if self.vel_add:
                add = add + bvel
            
            #add velocites relative to pelvis velocity
            if self.rel_vel_add:
                rel_vel = bvel
                for i in range(len(rel_vel)):
                    if i%2==0:
                        rel_vel[i] -= pvel[0]
                    else:
                        rel_vel[i] -= pvel[1]
                add = add + rel_vel
                
            return add
        
        #adds acceleration of center of mass, pelvis and joints
        def add_acc():
            acc = [(self.current_state[i]-last[i])/self.stepsize for i in range(self.STATE_V_PELVIS,self.STATE_V_PELVIS+3)]
            acc += [(self.current_state[i]-last[i])/self.stepsize for i in range(self.STATE_V_JOINTS,self.STATE_V_JOINTS+6)]
            acc += [(self.current_state[i]-last[i])/self.stepsize for i in range(self.STATE_CENTER_MASS_X+2,self.STATE_CENTER_MASS_X+4)]
            return acc
        
        #TODO: add additional obstacle observation
                      
        #rigid adjustments to normalize the observation values
        def adjust_manually():
            #joint angles and velocities
            for i in range(self.STATE_JOINTS,self.STATE_V_JOINTS):
                obs[i] /= 3.
            for i in range(self.STATE_V_JOINTS,self.STATE_CENTER_MASS_X):
                obs[i] /= 18.
            
            #pelvis rotation
            obs[0] /= 2.
            obs[self.STATE_V_PELVIS] /= 4.
            
            #pelvis velocity
            obs[self.STATE_V_PELVIS+1] /= 3.
            obs[self.STATE_V_PELVIS+2] /= 3.
            
            #added relative velocities
            #TODO: generalize for the case of vel and acc
            for i in range(self.STATE_OBSTACLE+3, self.ninput):
                obs[i] /= 8. if (i%2 == 1) else 80.
        
        #further normalization
        def normalize_large_input(v):
            for i in range(len(v)):
                if v[i] > 1: v[i] = np.sqrt(v[i])
                if v[i] < -1: v[i] = -np.sqrt(-v[i])
        
        #### call observation processing in order ####
        
        if self.vel_add or self.rel_vel_add:
            obs = obs + add_bodyPart_vel()
        if self.acc_add:
            obs = obs + add_acc()
        pelvis_relative()
        
        adjust_manually()
        normalize_large_input(obs)
        
        self.current_state = obs
        return self.current_state
    
    ################
    #### REWARD ####
    ################
    
    def compute_lig_pen(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2
            
        return lig_pen
    
    #lowering the pelvis results in penalty [y_threshold,y_done] -> [0,1]
    def compute_pelvisY_pen(self):
        threshold = self.PELVIS_Y_Initial - 0.1
        if self.current_state[self.STATE_PELVIS_Y] < threshold:
            norm = (self.current_state[self.STATE_PELVIS_Y] - self.PELVIS_Y_DONE)/(threshold-self.PELVIS_Y_DONE)
            return 1. - norm
        else:
            return 0
        
    #leaning of the upperbody
    def compute_posture_pen(self):
        hx = self.current_state[self.STATE_HEAD_X]
        if hx < 0.:                                                         #backward leaning
            return hx
        elif self.current_state[self.STATE_HEAD_Y] < self.hy_initial * 0.8: #forward leaning (starting at a certain head height)
            return -hx
        else:
            return 0.

    def joint_pen(self):
        #if the knee is pushed back, the angle is positive 
        angle_right_knee = self.current_state[self.STATE_KNEE_R]
        angle_left_knee = self.current_state[self.STATE_KNEE_L]
        pen = sum([max(0,a) for a in [angle_right_knee,angle_left_knee]])
        return pen
        
    def delta_x(self):
        return self.pelvis_x - self.prev_pelvis_x        

    def compute_reward(self):      
        reward=0.0
        
        if h.bitAt(self.rewardMode,1):
            reward += self.delta_x()
        if h.bitAt(self.rewardMode,2):
            reward -= self.compute_pelvisY_pen()**4 * 1e-3
        if h.bitAt(self.rewardMode,3):
            reward += self.compute_posture_pen() * 1e-2
        if h.bitAt(self.rewardMode,4):
            reward -= self.joint_pen() * 4e-2
        if h.bitAt(self.rewardMode,5):
            reward -= math.sqrt(self.compute_lig_pen()) * 10e-8
        
        return reward
    
    #########################
    #### Testing/Logging ####
    #########################
    
    def getDistanceTravelled(self):
        return self.pelvis_x - self.initital_pelvis_x


    
#testing
if __name__=='__main__':
    
    env = diffEnv(visualize=False, difficulty=0, redundant=False, vel_add=False, rel_vel_add=True, acc_add=False)
    env.reset()
    
    obs = env.get_observation()
    
    print(len(obs))
    print(env.ninput)
    for i in range(env.ninput):
        print(str(i) + ": " + str(obs[i]))