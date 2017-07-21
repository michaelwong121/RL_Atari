import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import reinforce
from utils import *
from atari_wrappers import *

class ValueEstimator ():
    def __init__(self,
                input_shape,
                num_actions,
                learning_rate = 1e-4,
                clip_var = 10,
                scope="value_estimator"):
                    
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.uint8, [None] + list(input_shape))
            self.target = tf.placeholder(tf.float32)
            self.learning_rate = tf.placeholder(tf.float32)
            
            casted_input = tf.cast(self.state,   tf.float32) / 255.0
            
            #TODO fill in the estimator network
            hid1 = layers.convolution2d(casted_input, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            hid2 = layers.convolution2d(hid1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            hid3 = layers.convolution2d(hid2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            hid3_flat = layers.flatten(hid3)
            hid4 = layers.fully_connected(hid3_flat, num_outputs=512,         activation_fn=tf.nn.relu)
            self.output = layers.fully_connected(hid4, num_outputs=num_actions, activation_fn=None)
            
            
            self.loss = tf.squared_difference(self.output, self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            #self.train_fn = tf.clip_by_norm(self.optimizer.minimize(self.loss), clip_var)
            #TODO add clipping to improve performance?
            self.train_fn = self.optimizer.minimize(self.loss)
    
    def predict(self, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, {self.state: state})
        
    def update(self, state, target, sess = None):
        sess = sess or tf.get_default_session()
        sess.run(self.train_fn, feed_dict={self.state: state, 
                                           self.target: target,
                                           self.learning_rate: 1e-4})
        

class PolicyEstimator ():
    def __init__(self,
                input_shape,
                num_actions,
                learning_rate = 1e-4,
                clip_var = 10,
                scope="policy_estimator"):
                    
        with tf.variable_scope(scope):
            self.state  = tf.placeholder(tf.uint8,      [None] + list(input_shape))
            self.action = tf.placeholder(tf.int32,      [None])
            self.advantage = tf.placeholder(tf.float32)
            self.learning_rate = tf.placeholder(tf.float32)
            
            casted_input = tf.cast(self.state,   tf.float32) / 255.0
            
            #TODO change the estimator network
            hid1 = layers.convolution2d(casted_input, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            hid2 = layers.convolution2d(hid1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            hid3 = layers.convolution2d(hid2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            hid3_flat = layers.flatten(hid3)
            hid4 = layers.fully_connected(hid3_flat, num_outputs=512,         activation_fn=tf.nn.relu)
            hid5 = layers.fully_connected(hid4, num_outputs=num_actions, activation_fn=None)
            
            self.output = tf.nn.softmax(hid5)
            
            self.action_prob = tf.map_fn(lambda x: tf.gather(x[0], x[1]), 
                                                  (self.output, self.action),
                                                  dtype = tf.float32)
            
            self.loss = -tf.log(self.action_prob) * self.advantage
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            #self.train_fn = tf.clip_by_norm(self.optimizer.minimize(self.loss), clip_var)
            #TODO add clipping to improve performance?
            self.train_fn = self.optimizer.minimize(self.loss)
    
    def predict(self, s, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, {self.state: s})
        
    def update(self, state, action, advantage, sess = None):
        sess = sess or tf.get_default_session()
        sess.run(self.train_fn, feed_dict={self.state: state, 
                                           self.action: action, 
                                           self.advantage: advantage,
                                           self.learning_rate: 1e-4})

def atari_learn(env,
                session,
                V,
                pi,
                num_timesteps):
    # This is just a rough estimate
    # TODO might not need to divide by 4?
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

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    reinforce.learn(
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

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    
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
    
    """
    state = env.reset()
    pi.predict([state], session)[0]
    """
    atari_learn(env, session, V, pi, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
