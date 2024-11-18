from environments.basic import BasicDoomEnv
from agents.dqn_agent import DQNAgent

if __name__ == "__main__":
    env = BasicDoomEnv()
    agent = DQNAgent(env)
    agent.train(timesteps=100000)
    agent.save("models/dqn_basic")
    env.close()
