import gym
import atexit

class Game:

    def __init__(self) -> None:
        self.env = gym.make('CartPole-v0')
        self.reset()
        atexit.register(self.close)
    
    def reset(self):
        observation = self.env.reset()
        self.env.render()
        return observation

    def play_action(self, action):
        # print('action is :', type(action), action)
        # action = action.item()
        observation, reward, done, info = self.env.step(action)
        self.env.render()
        return observation, reward, done, info
    
    def close(self):
        self.env.close()
