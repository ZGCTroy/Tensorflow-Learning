import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
env.reset()
reward_sum = 0

# while random_episodes < 10:
