from evaluate import evaluate_hands

def calculate_reward(env, player, action, current_winrate):
    opponent = 1 - player
    invested = env.total_bet_ai1 if player == 0 else env.total_bet_ai2

    if action == 0:
        if current_winrate < 0.3:
            return +0.2  # good fold
        else:
            return -round(current_winrate * invested, 2)

    if env.done and env.round_stage == "river":
        result = evaluate_hands(env.ai1_hand, env.ai2_hand, env.board)
        if player == 0:
            if result == 1: return +env.pot / 2
            elif result == -1: return -invested
            else: return 0
        else:
            if result == -1: return +env.pot / 2
            elif result == 1: return -invested
            else: return 0

    return round((current_winrate - env.last_winrate) * env.pot, 3)
