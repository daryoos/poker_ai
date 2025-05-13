from treys import Evaluator, Card
from cards import convert_to_treys
import random
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

def get_winrate_pypokerengine(ai_hand, board, nb_simulation=25):
    hole = [to_ppe(c) for c in ai_hand]
    community = [to_ppe(c) for c in board]

    win_rate = estimate_hole_card_win_rate(
        nb_simulation=nb_simulation,
        nb_player=2,
        hole_card=gen_cards(hole),
        community_card=gen_cards(community)
    )
    return round(win_rate, 3)

def to_ppe(card):
    rank_map = {
        14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
         9: '9',  8: '8',  7: '7',  6: '6',  5: '5',
         4: '4',  3: '3',  2: '2'
    }
    suit_map = {'s': 'S', 'h': 'H', 'd': 'D', 'c': 'C'}
    rank = rank_map[card[0]]
    suit = suit_map[card[1]]
    return suit + rank  # e.g., 'HA'

def evaluate_preflop_hand_strength(card1, card2, nb_simulation=200):
    hole = [to_ppe(card1), to_ppe(card2)]
    win = 0

    for _ in range(nb_simulation):
        deck = [s + r for r in "23456789TJQKA" for s in "SHDC"]
        used = set(hole)
        deck = [c for c in deck if c not in used]
        opp_hand = random.sample(deck, 2)
        used.update(opp_hand)
        deck = [c for c in deck if c not in used]
        community = random.sample(deck, 5)

        ai_score = HandEvaluator.eval_hand(gen_cards(hole), gen_cards(community))
        opp_score = HandEvaluator.eval_hand(gen_cards(opp_hand), gen_cards(community))

        if ai_score < opp_score:
            win += 1
        elif ai_score == opp_score:
            win += 0.5

    return round(win / nb_simulation, 3)

def get_hand_category(card1, card2):
    if card1[0] == card2[0]: return 0
    elif card1[1] == card2[1] and abs(card1[0] - card2[0]) == 1: return 1
    elif card1[1] == card2[1]: return 2
    elif abs(card1[0] - card2[0]) == 1: return 3
    elif max(card1[0], card2[0]) >= 13: return 4
    elif min(card1[0], card2[0]) <= 6: return 5
    else: return 6

def has_flush_draw(hand, board):
    suits = [c[1] for c in hand + board]
    return int(any(suits.count(suit) >= 4 for suit in 'shdc'))

def has_straight_draw(hand, board):
    ranks = sorted(set([c[0] for c in hand + board]))
    for i in range(len(ranks) - 3):
        if ranks[i+3] - ranks[i] <= 4: return 1
    return 0

def get_overcards_count(hand, board):
    if not board: return 0
    max_board = max(c[0] for c in board)
    return sum(c[0] > max_board for c in hand)

def get_hand_strength(hand, board):
    if len(board) < 3:
        # Preflop: use estimated strength
        return 1.0 - evaluate_preflop_hand_strength(hand[0], hand[1])  # Normalize: lower is better

    evaluator = Evaluator()
    h = [Card.new(convert_to_treys(c)) for c in hand]
    b = [Card.new(convert_to_treys(c)) for c in board]
    score = evaluator.evaluate(b, h)
    return score / 7462  # normalized

def get_hand_strength_bucket(score):
    if score < 0.2: return 4
    elif score < 0.4: return 3
    elif score < 0.6: return 2
    elif score < 0.8: return 1
    else: return 0

def get_aggression_factor(raise_count, call_count):
    return round(raise_count / (call_count + 1), 2)

def get_betting_pattern_index(last_action_ai1, last_action_ai2):
    return last_action_ai1 * 3 + last_action_ai2

def get_full_state(hand, board, position, pot_size, stack_ai, stack_opponent,
                   current_bet, round_stage, winrate,
                   last_action_ai1, last_action_ai2,
                   total_bet_opp, raises_this_street,
                   fold_count, call_count, raise_count,
                   total_hands, action_history):

    card1_val, card2_val = hand[0][0], hand[1][0]
    same_suit = int(hand[0][1] == hand[1][1])
    hand_cat = get_hand_category(hand[0], hand[1])
    street = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[round_stage]
    spr = min(stack_ai, stack_opponent) / (pot_size + 1e-6)

    flush_draw = has_flush_draw(hand, board)
    straight_draw = has_straight_draw(hand, board)
    overcards = get_overcards_count(hand, board)

    score = get_hand_strength(hand, board)
    bucket = get_hand_strength_bucket(score)

    opponent_aggression = get_aggression_factor(raise_count, call_count)
    pattern_idx = get_betting_pattern_index(last_action_ai1, last_action_ai2)

    total_hands = max(total_hands, 1)

    # Action history embedding
    action_sequence = [0.0] * 10  # neutral padding (valid in Box[0,1])

    flat = [a for (_, a) in action_history[-10:]]
    for i in range(len(flat)):
        action_sequence[i] = flat[i] / 4.0  # normalized action (0–1 range)

    opponent_looseness = (call_count + raise_count) / total_hands
    opponent_fold_rate = fold_count / total_hands

    return [
        card1_val / 14,
        card2_val / 14,
        same_suit,
        hand_cat,
        position,
        stack_ai / 100,
        stack_opponent / 100,
        pot_size / 100,
        current_bet / 100,
        spr,
        winrate,
        score,
        flush_draw,
        straight_draw,
        overcards,
        last_action_ai2,
        total_bet_opp / 100,
        raises_this_street / 3,
        fold_count / total_hands,
        call_count / total_hands,
        raise_count / total_hands,
        bucket,
        street,
        opponent_aggression,
        opponent_looseness,
        opponent_fold_rate,
        pattern_idx
    ] + action_sequence
