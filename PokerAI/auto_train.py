import subprocess
import pandas as pd
import os
import time
from config import get_current_generation

# === Settings ===
MAX_GENERATIONS = 20
MIN_WINRATE = 0.52
WAIT_BETWEEN_GENERATIONS = 5  # seconds
METRICS_FILE = "metrics.csv"

def read_last_winrate():
    if not os.path.exists(METRICS_FILE):
        return None
    try:
        df = pd.read_csv(METRICS_FILE)
        if len(df) == 0 or "winrate_vs_prev" not in df.columns:
            return None
        return df["winrate_vs_prev"].iloc[-1]
    except pd.errors.EmptyDataError:
        return None

def main():
    while True:
        current_gen = get_current_generation()
        print(f"\n🚀 Starting training for Generation {current_gen}")

        # Stop if generation cap is reached
        if current_gen >= MAX_GENERATIONS:
            print("✅ Reached max generation limit.")
            break

        # Early stopping if winrate fell below threshold
        last_winrate = read_last_winrate()
        if current_gen >= 3 and last_winrate is not None and last_winrate < MIN_WINRATE:
            print(f"🛑 Stopping: last winrate ({last_winrate:.3f}) dropped below {MIN_WINRATE}")
            break

        # Call the training script
        try:
            subprocess.run(["python", "train.py"], check=True)
        except subprocess.CalledProcessError as e:
            print("🔥 Training failed. Exiting loop.")
            print(e)
            break

        # Optional pause between generations
        time.sleep(WAIT_BETWEEN_GENERATIONS)

if __name__ == "__main__":
    main()
