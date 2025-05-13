from stable_baselines3 import PPO
from environment import Environment
import numpy as np

def evaluate_models(model_path_new, model_path_old, num_episodes=1000):
    wins = 0
    losses = 0
    ties = 0

    env = Environment()
    env.frozen_opponent = PPO.load(model_path_old)
    model_new = PPO.load(model_path_new)

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            if env.current_player == 0:
                action, _ = model_new.predict(obs, deterministic=True)
            else:
                action, _ = env.frozen_opponent.predict(obs, deterministic=True)

            obs, reward, done, _, _ = env.step(action)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            ties += 1

    winrate = wins / num_episodes
    elo_change = calculate_elo_change(winrate)

    return winrate, elo_change


def calculate_elo_change(winrate, k=32):
    expected = 0.5
    return k * (winrate - expected)
