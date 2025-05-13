from stable_baselines3 import PPO
from environment import Environment
from config import ppo_gen

# === CONFIG ===
MODEL_PATH = f"poker_ppo_gen{ppo_gen - 1}"  # ← Change this to test different versions
NUM_HANDS = 5

# Load trained model
model = PPO.load(MODEL_PATH)
env = Environment()
env.frozen_opponent = model  # Both agents use the same model (mirror match)

# Run hands
for hand_index in range(NUM_HANDS):
    print(f"\n=== HAND {hand_index + 1} START ===")
    obs, _ = env.reset()
    done = False
    step = 1

    print("AI_1 Hand:", env.ai1_hand)
    print("AI_2 Hand:", env.ai2_hand)

    while not done:
        current_player = env.current_player
        player_name = "AI_1" if current_player == 0 else "AI_2"

        print(f"\n--- Step {step} | {player_name}'s turn ---")
        print("Round:", env.round_stage)
        print("Board:", env.board)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if "winrate" in info:
            print("Winrate estimate:", round(info["winrate"] * 100, 2), "%")

        step += 1

    print("\n=== HAND END ===")
    print("Final Reward for AI_1:", reward)
    print("Final Board:", info['board'])
    print("AI_1 Hand:", info['ai1_hand'])
    print("AI_2 Hand:", info['ai2_hand'])
