import gym
import numpy as np
import random
import atexit
import os

MODEL_FILE = 'data[Long].csv'
LEARN = True # loads model if False

# create Taxi environment
env = gym.make('Taxi-v3')

# Q-Learning
state_size = env.observation_space.n  # total number of states (S)
action_size = env.action_space.n      # total number of actions (A)

if (not LEARN) and os.path.exists(MODEL_FILE):
    qtable = np.loadtxt(MODEL_FILE, delimiter=',')
else:
    qtable = np.zeros((state_size, action_size))
# qtable = np.zeros((state_size, action_size))

learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0     # probability that our agent will explore
decay_rate = 0.005 # of epsilon


def save():
    np.savetxt(MODEL_FILE, qtable, delimiter=',')


# atexit.register(save)
# create a new instance of taxi, and get the initial state

num_episode = 2000
num_steps = 99

episode= 0
for episode in range(num_episode):
    state = env.reset()
    done = False
    score = 0
    for step in range(num_steps):

        # sample a random action from the list of available actions
        if LEARN and random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])
            

        # perform this action on the environment
        new_state, reward, done, info = env.step(action)

        if LEARN:
            qtable[state, action] += learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state,action])
        # print the new state
        state = new_state
        score += reward
        if not LEARN:
            pass
            # env.render()
            # input()
        if done:
            break
    epsilon = np.exp(-decay_rate*episode)
    episode += 1
    # if not LEARN:
    print('Score:', score)
        # input()

# end this instance of the taxi environment
env.close()

save() 
