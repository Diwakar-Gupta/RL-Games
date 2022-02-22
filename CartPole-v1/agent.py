import torch
import random
import numpy as np
from environment import Game
from collections import deque
from model import Linear_QNet, QTrainer
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)

        self.model = Linear_QNet(4, 256, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)

        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, 'model.pth')
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name))
            self.model.eval()
    
    def get_state(self, game, observation):
        state = [
            observation,
        ]

        return np.array(state)

    def train_sort_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # return random.randint(0, 1)
        final_move = 0
        if random.randint(0, 200) < 28:
            move = random.randint(0, 1)
            final_move = move
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move
        
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


def train():

    agent = Agent()
    game = Game()
    record = 0
    score = 0
    observation = game.reset()
    # observation, reward, done
    
    while True:
        oldstate = observation
        final_move = agent.get_action(oldstate)

        # print('type of final', final_move, torch.argmax(final_move).item())
        observation, reward, done, info = game.play_action(final_move)

        if reward > 0:
            reward = 1
        else:
            reward = -50
        

        agent.get_state(game, observation)
        agent.train_sort_memory(oldstate, final_move, reward, observation, done)
        agent.remember(oldstate, final_move, reward, observation, done)
        score += reward

        # print(observation, reward, done, info)
        if done:
            observation = game.reset()
            agent.n_games += 1
        
            agent.train_long_memory()
            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            score = 0


if __name__ == '__main__':
    train()
