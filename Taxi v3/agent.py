import gym
import numpy as np
import random
import os
import time


# create Taxi environment
env = gym.make('Taxi-v3')

# Q-Learning
state_size = env.observation_space.n  # total number of states (S)
action_size = env.action_space.n      # total number of actions (A)
qtable = np.zeros((state_size, action_size))

learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0     # probability that our agent will explore
decay_rate = 0.005 # of epsilon


# create a new instance of taxi, and get the initial state

max_steps = 99

episode = 1

total_score = 0
record = 0

def prnt(action, reward, step):
    os.system('clear')
    env.render()
    print(action)
    print()
    print('Episodes:', episode-1)
    print('TimeStep:', step)
    print('Reward:', reward)
    print('Average:', total_score//episode)


# for episode in range(num_episode):
while True:
    state = env.reset()
    done = False
    score = 0

    for step in range(max_steps):

        # sample a random action from the list of available actions
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])
            

        # perform this action on the environment
        new_state, reward, done, info = env.step(action)

        qtable[state, action] += learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state,action])
        # print the new state
        state = new_state
        score += reward
        
        # env.render()
        prnt(action, reward, step)
        # input()
        if done:
            break
        # time.sleep(0.9)
    epsilon = np.exp(-decay_rate*episode)
    episode += 1

    record = max(record, score)
    total_score += score
