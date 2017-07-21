import sys
import argparse
import gym
from gym import wrappers
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
import os.path as osp
import distutils.spawn

#distutils.spawn.find_executable('ffmpeg') is not None


from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet2"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value2"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


env = gym.make('PongNoFrameskip-v4')

q_function = atari_model
frame_history_len = 4
gamma = 0.99
sess = tf.Session()

if len(env.observation_space.shape) == 1:
    # This means we are running on low-dimensional observations (e.g. RAM)
    input_shape = env.observation_space.shape
else:
    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)
num_actions = env.action_space.n

# set up placeholders
# placeholder for current observation (or state)
obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
# placeholder for current action
act_t_ph              = tf.placeholder(tf.int32,   [None])
# placeholder for current reward
rew_t_ph              = tf.placeholder(tf.float32, [None])
# placeholder for next observation (or state)
obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
# placeholder for end of episode mask
# this value is 1 if the next state corresponds to the end of an episode,
# in which case there is no Q-value at the next state; at the end of an
# episode, only the current state reward contributes to the target, not the
# next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
done_mask_ph          = tf.placeholder(tf.float32, [None])

# casting to float on GPU ensures lower data transfer times.
obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0


# set up Q function network
q_val = q_function(obs_t_float, num_actions, scope="q_func2", reuse=False)
q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func2')

# set up target Q network
target_q_val = q_function(obs_t_float, num_actions, scope="target_q_func2", reuse=False)
target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func2')

# calculate target Q value and total error
max_target_Q = tf.reduce_max(target_q_val, reduction_indices=[1])
done_bool = tf.equal(done_mask_ph, tf.constant(1.0))
y = tf.where(done_bool, rew_t_ph, rew_t_ph + gamma * max_target_Q)

this_q_val = tf.map_fn(lambda x: tf.gather(x[0], x[1]), 
                                  (q_val, act_t_ph),
                                  dtype = tf.float32)

total_error = tf.square(y - this_q_val)

obs = env.reset()
