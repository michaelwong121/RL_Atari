import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import reinforce_episodic
from utils import *
from atari_wrappers import *

class ValueEstimator ():
    def __init__(self,
                input_shape,
                num_actions,
                #learning_rate = 1e-7,
                clip_var = 10,
                scope="value_estimator"):
                    
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None] + list(input_shape))
            self.target = tf.placeholder(tf.float32)
            #self.learning_rate = tf.placeholder(tf.float32)
            
            
            casted_input = self.state / 255.0
            """
            #TODO fill in the estimator network
            hid1 = layers.fully_connected(casted_input, num_outputs=256, activation_fn=tf.nn.relu)
            hid2 = layers.fully_connected(hid1,         num_outputs=128, activation_fn=tf.nn.relu)
            hid3 = layers.fully_connected(hid2,         num_outputs=64,  activation_fn=tf.nn.relu)
            self.output = layers.fully_connected(hid3, num_outputs=num_actions, activation_fn=None)
            """
            
            self.output = layers.fully_connected(casted_input, num_outputs=1, activation_fn=None)
            
            self.loss = tf.squared_difference(self.output, self.target)
            self.sum_loss = tf.reduce_sum(self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            #self.train_fn = tf.clip_by_norm(self.optimizer.minimize(self.loss), clip_var)
            #TODO add clipping to improve performance?
            self.train_fn = self.optimizer.minimize(self.loss)
    
    def predict(self, state, sess):
        return sess.run(self.output, {self.state: [state]})
        
    def update(self, state, target, sess):
        _, loss = sess.run([self.train_fn, self.sum_loss], 
                            feed_dict={self.state: [state], 
                                       self.target: target})
        return loss
        

class PolicyEstimator ():
    def __init__(self,
                input_shape,
                num_actions,
                #learning_rate = 1e-4,
                clip_var = 10,
                scope="policy_estimator"):
                    
        with tf.variable_scope(scope):
            self.state  = tf.placeholder(tf.float32,    [None] + list(input_shape))
            self.action = tf.placeholder(tf.int32)
            self.advantage = tf.placeholder(tf.float32)
            #self.learning_rate = tf.placeholder(tf.float32)
            
            casted_input = self.state / 255.0 # normalize
            
            #TODO change the estimator network
            #hid1 = layers.fully_connected(casted_input, num_outputs=256, activation_fn=tf.nn.relu)
            """
            hid2 = layers.fully_connected(hid1,         num_outputs=128, activation_fn=tf.nn.relu)
            hid3 = layers.fully_connected(hid2,         num_outputs=64,  activation_fn=tf.nn.relu)
            hid4 = layers.fully_connected(hid3, num_outputs=num_actions, activation_fn=None)
            """
            hid4 = layers.fully_connected(casted_input, num_outputs=num_actions, activation_fn=None)
            
            self.output = tf.squeeze(tf.nn.softmax(hid4))
            
            self.action_prob = tf.gather(self.output, self.action)
            
            self.loss = -tf.log(self.action_prob) * self.advantage
            self.sum_loss = tf.reduce_sum(self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)
            #self.train_fn = tf.clip_by_norm(self.optimizer.minimize(self.loss), clip_var)
            #TODO add clipping to improve performance?
            self.train_fn = self.optimizer.minimize(self.loss)
    
    def predict(self, s, sess):
        return sess.run(self.output, {self.state: [s]})
        
    def update(self, state, action, advantage, sess):
        _, loss = sess.run([self.train_fn, self.sum_loss], 
                           feed_dict={self.state: np.array([state]), 
                                      self.action: np.array(action), 
                                      self.advantage: np.array(advantage)})
        return loss

def atari_learn(env,
                session,
                V,
                pi,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    # TODO right now just use constant learning rate
    """
    lr_multiplier = 1.0 
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )
    """

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps


    reinforce_episodic.learn(
                    env,
                    #optimizer_spec,
                    session,
                    V,
                    pi,
                    stopping_criterion=None,
                    gamma=0.99,
                    #grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(seed):
    env = gym.make('Pong-ram-v0')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind_ram(env)

    return env

def main():
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(seed)
    session = get_session()
    
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, img_c)
    num_actions = env.action_space.n

    V = ValueEstimator(input_shape, num_actions)
    pi = PolicyEstimator(input_shape, num_actions)
    
    session.run(tf.global_variables_initializer())
    
    atari_learn(env, session, V, pi, num_timesteps=int(4e7))

if __name__ == "__main__":
    main()
