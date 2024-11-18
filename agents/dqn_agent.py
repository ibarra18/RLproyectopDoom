from stable_baselines3 import DQN

class DQNAgent:
    def __init__(self, env):
        self.model = DQN('CnnPolicy', env, verbose=1)

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = DQN.load(path)
