import os

GEN_FILE = "current_gen.txt"

def get_current_generation():
    if not os.path.exists(GEN_FILE):
        with open(GEN_FILE, "w") as f:
            f.write("0")
    with open(GEN_FILE, "r") as f:
        return int(f.read().strip())

def increment_generation():
    gen = get_current_generation() + 1
    with open(GEN_FILE, "w") as f:
        f.write(str(gen))
    return gen

# Global variable you can import
ppo_gen = get_current_generation()
