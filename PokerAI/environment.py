import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3 import PPO
import os
from config import ppo_gen, increment_generation

gen_path = f"poker_ppo_gen{ppo_gen - 1}"
fallback_path = "poker_ppo_model"

# Game logic
from cards import create_deck, deal_hole_cards, burn_card
from features import get_full_state, get_winrate_pypokerengine, evaluate_preflop_hand_strength
from reward import calculate_reward
from evaluate import evaluate_hands

class Environment(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(37,),  # Update if you add/remove features in get_full_state
            dtype=np.float32
        )
        # 0: fold, 1: call, 2: raise (min), 3: raise (pot), 4: all-in
        self.action_space = spaces.Discrete(5)

        if os.path.exists(gen_path + ".zip") or os.path.isdir(gen_path):
            self.frozen_opponent = PPO.load(gen_path)
            print(f"[INFO] Loaded frozen opponent from {gen_path}")
        elif os.path.exists(fallback_path + ".zip") or os.path.isdir(fallback_path):
            self.frozen_opponent = PPO.load(fallback_path)
            print(f"[INFO] Loaded frozen opponent from {fallback_path}")
        else:
            self.frozen_opponent = None
            print("[INFO] No frozen opponent model found — using random actions.")

        self.reset_vars()

    def reset_vars(self):
        self.deck = []
        self.ai1_hand = []
        self.ai2_hand = []
        self.board = []
        self.position = 0
        self.round_stage = 'preflop'
        self.done = False
        self.current_player = 0
        self.last_winrate = 0.0
        self.reward_given = False
        self.last_action_ai1 = 1
        self.last_action_ai2 = 1
        self.total_bet_ai1 = 0
        self.total_bet_ai2 = 0
        self.raises_this_street = 0
        self.fold_count = 0
        self.call_count = 0
        self.raise_count = 0
        self.action_history = [] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.reset_vars()

        self.deck = create_deck()
        self.ai1_hand, self.ai2_hand, self.deck = deal_hole_cards(self.deck)
        self.board = []
        self.round_stage = 'preflop'
        self.current_player = 0

        self.pot = 3
        self.stack_ai1 = 100
        self.stack_ai2 = 100
        self.current_bet = 2

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        hand = self.ai1_hand if self.current_player == 0 else self.ai2_hand
        board = self.board.copy()
        position = int(self.current_player == 1)  # Acts last = 1
        opponent_last_action = self.last_action_ai2 if self.current_player == 0 else self.last_action_ai1
        opponent_total_bet = self.total_bet_ai2 if self.current_player == 0 else self.total_bet_ai1
        total_hands = max(1, self.fold_count + self.call_count + self.raise_count)

        state = get_full_state(
            hand, board, position, self.pot,
            self.stack_ai1, self.stack_ai2,
            self.current_bet, self.round_stage,
            self.last_winrate,
            self.last_action_ai1, self.last_action_ai2,
            opponent_total_bet, self.raises_this_street,
            self.fold_count, self.call_count, self.raise_count,
            total_hands, self.action_history 
        )

        obs = np.array(state, dtype=np.float32)
        # print(f"[DEBUG] obs shape: {obs.shape}, values: {obs}")
        # print(f"[DEBUG] expected shape: {self.observation_space.shape}")
        # assert self.observation_space.contains(obs), "Observation out of bounds!"
        return obs
    
    def get_raise_amount(self, action):
        if action == 2:
            return min(4, self.stack_ai1)  # Min raise
        elif action == 3:
            return min(self.pot, self.stack_ai1)  # Pot-sized
        elif action == 4:
            return self.stack_ai1  # All-in
        else:
            return self.current_bet if action == 1 else 0  # Call or fold

    def step(self, action):
        info = {
            "ai1_hand": self.ai1_hand,
            "ai2_hand": self.ai2_hand,
            "board": self.board.copy(),
            "round_stage": self.round_stage,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "stack_ai1": self.stack_ai1,
            "stack_ai2": self.stack_ai2,
            "last_action_ai1": self.last_action_ai1,
            "last_action_ai2": self.last_action_ai2,
            "total_bet_ai1": self.total_bet_ai1,
            "total_bet_ai2": self.total_bet_ai2,
            "fold_count": self.fold_count,
            "call_count": self.call_count,
            "raise_count": self.raise_count,
            "current_player": self.current_player,
            "winrate": self.last_winrate,
        }

        # Opponent (AI2)
        if self.current_player == 1:
            obs = self._get_obs()
            if self.frozen_opponent:
                action, _ = self.frozen_opponent.predict(obs, deterministic=False)
            else:
                action = self.action_space.sample()

            if action == 0:
                self.fold_count += 1
            elif action == 1:
                self.call_count += 1
            elif action in [2, 3, 4]:
                self.raise_count += 1

        # Get bet amount if action is raise
        raise_amount = self.get_raise_amount(action)

        # Track actions and update stack/pot
        if self.current_player == 0:
            self.last_action_ai1 = action
            if action in [1, 2, 3, 4]:
                self.total_bet_ai1 += raise_amount
                self.stack_ai1 -= raise_amount
                self.pot += raise_amount
        else:
            self.last_action_ai2 = action
            if action in [1, 2, 3, 4]:
                self.total_bet_ai2 += raise_amount
                self.stack_ai2 -= raise_amount
                self.pot += raise_amount

        if action in [2, 3, 4]:
            self.raises_this_street += 1

        # Winrate evaluation
        hand = self.ai1_hand if self.current_player == 0 else self.ai2_hand
        if self.round_stage != "preflop":
            current_winrate = get_winrate_pypokerengine(hand, self.board)
        else:
            current_winrate = evaluate_preflop_hand_strength(hand[0], hand[1])
        self.last_winrate = current_winrate

        # Reward logic
        reward = calculate_reward(self, self.current_player, action, current_winrate)

        if action == 0:
            self.done = True
            return self._get_obs(), round(reward, 3), True, False, info

        self.action_history.append((self.current_player, action))
        if len(self.action_history) > 10:
            self.action_history.pop(0)

        # Advance round
        if self.round_stage == "preflop":
            self.advance_to_flop()
        elif self.round_stage == "flop":
            self.advance_to_turn()
        elif self.round_stage == "turn":
            self.advance_to_river()
        elif self.round_stage == "river":
            score1 = evaluate_hands(self.ai1_hand, self.ai2_hand, self.board)
            score2 = evaluate_hands(self.ai2_hand, self.ai1_hand, self.board)
            reward = 1 if score1 < score2 else -1 if score1 > score2 else 0
            self.done = True
            return self._get_obs(), reward, True, False, info

        self.current_player = 1 - self.current_player
        return self._get_obs(), reward, False, False, info

    def advance_to_flop(self):
        burn_card(self.deck)
        self.board.extend([self.deck.pop(), self.deck.pop(), self.deck.pop()])
        self.round_stage = "flop"
        self.raises_this_street = 0

    def advance_to_turn(self):
        burn_card(self.deck)
        self.board.append(self.deck.pop())
        self.round_stage = "turn"
        self.raises_this_street = 0

    def advance_to_river(self):
        burn_card(self.deck)
        self.board.append(self.deck.pop())
        self.round_stage = "river"
        self.raises_this_street = 0
