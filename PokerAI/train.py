from stable_baselines3 import PPO
from environment import Environment
from stable_baselines3.common.callbacks import ProgressBarCallback, CheckpointCallback
from config import ppo_gen, increment_generation
from model_evaluation import evaluate_models
import pandas as pd
import os

# Config
total_timesteps = 2_000_000
model_path = f"poker_ppo_gen{ppo_gen}"
prev_model_path = f"poker_ppo_gen{ppo_gen - 1}"

# Setup environment
print(f"Training Generation {ppo_gen}")
env = Environment()

# Model architecture
policy_kwargs = dict(net_arch=[256, 256])
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs",
    policy_kwargs=policy_kwargs
)

# Add checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,  # Save every 100k timesteps
    save_path=f"./checkpoints/gen{ppo_gen}",
    name_prefix="model"
)

# Train model
model.learn(
    total_timesteps=total_timesteps,
    callback=[ProgressBarCallback(), checkpoint_callback]
)

# Save final model
model.save(model_path)
print(f"Saved final model: {model_path}")

# Evaluation vs previous generation
if ppo_gen > 0 and os.path.exists(prev_model_path + ".zip"):
    winrate, elo_change = evaluate_models(model_path, prev_model_path)
    print(f"Winrate vs Gen {ppo_gen - 1}: {winrate:.3f}, Elo Δ: {elo_change:+.2f}")

    # Log to metrics.csv
    row = {
        "generation": ppo_gen,
        "winrate_vs_prev": winrate,
        "elo_change": elo_change
    }
    df = pd.DataFrame([row])
    if os.path.exists("metrics.csv"):
        df.to_csv("metrics.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("metrics.csv", index=False)

# Increment generation counter
increment_generation()
