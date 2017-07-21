import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
import collections
from utils import *


""" following the code in 
https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb
"""

def learn(env,
          #optimizer_spec,
          sess,
          V,
          pi,
          stopping_criterion=None,
          gamma=0.99,
          #grad_norm_clipping=10
          ):
    """
    
    env = gym.make('Pong-ram-v0')
    sess = tf.Session()
    
    
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, img_c)
    num_actions = env.action_space.n
    
    
    V = ValueEstimator(input_shape, num_actions)
    pi = PolicyEstimator(input_shape, num_actions)
    
    sess.run(tf.global_variables_initializer())
    
    stopping_criterion = None
    gamma = 0.99
    
    """

    MEMORY = 100
    LOG_EVERY_N_EPISODE = 500
    mean_reward      = -float('nan')
    mean_length      = -float('nan')
    track_returns = []
    track_length = []
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward"])
    
    
    for ep in range(100001):
        
        # check stopping criteria
        if stopping_criterion is not None and stopping_criterion(env):
            break
        
        state = env.reset()
        episode = []
        
        ep_length = 0 # not used
        G = 0 # total reward for this episode
        I = 1 # discount multiplier
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = pi.predict(state, sess)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(state=state, action=action, reward=reward))
    
            # Update culmulated reward and total discount rate
            G += reward * I
            I *= gamma
    
            if done:
                break
                
            state = next_state
    
        ep_length = t
        
        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(gamma**i * t.reward for i, t in enumerate(episode[t:]))
            
            # Update our value estimator
            V.update(transition.state, total_return, sess)
            # Calculate baseline/advantage
            baseline_value = V.predict(transition.state, sess)            
            advantage = total_return - baseline_value
            # Update our policy estimator
            pi.update(transition.state, transition.action, advantage, sess)
            
            """
            pi.update(transition.state, transition.action, total_return, sess)
            """
            
        track_returns.append(G)
        track_returns = track_returns[-MEMORY:]
        mean_return = np.mean(track_returns)
        track_length.append(ep_length)
        track_length = track_length[-MEMORY:]
        mean_length = np.mean(track_length)
        
        if ep % LOG_EVERY_N_EPISODE == 0:
            print("Episode %d" % (ep,))
            print("mean return %f" % mean_return)
            print("mean Policy length %d" % mean_length)
            #print("Value loss %d" % v_loss)
            sys.stdout.flush()
            
        
    






    
    
    
    
    

