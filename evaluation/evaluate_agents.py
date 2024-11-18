import argparse
from environments.basic import BasicDoomEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from agents.sac_agent import SACAgent

def evaluate(agent_class, model_path, env_class, episodes=10):
    env = env_class()
    agent = agent_class(env)
    agent.load(model_path)
    rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "ppo", "a2c", "sac"], required=True, help="Type of agent to evaluate")
    parser.add_argument("--env", choices=["basic", "defend", "health"], required=True, help="Environment to use")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    env_map = {
        "basic": BasicDoomEnv,
        # Agrega otras clases de entornos aquí
    }

    agent_map = {
        "dqn": DQNAgent,
        "ppo": PPOAgent,
        "a2c": A2CAgent,
        "sac": SACAgent,
    }

    env_class = env_map[args.env]
    agent_class = agent_map[args.agent]

    model_path = f"models/{args.agent}_{args.env}"
    rewards = evaluate(agent_class, model_path, env_class, episodes=args.episodes)

    print(f"Average reward over {args.episodes} episodes: {sum(rewards) / len(rewards)}")
