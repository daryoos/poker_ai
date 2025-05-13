import random
from treys import Card, Evaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator

suits = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs
ranks = list(range(2, 15))    # 2 - 14 (J=11, Q=12, K=13, A=14)

def create_deck():
    return [(rank, suit) for rank in ranks for suit in suits]

def deal_hole_cards(deck):
    random.shuffle(deck)
    ai_hand = [deck.pop(), deck.pop()]
    opponent_hand = [deck.pop(), deck.pop()]
    return ai_hand, opponent_hand, deck

def to_ppe(card):
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5',
                4: '4', 3: '3', 2: '2'}
    suit_map = {'s': 'S', 'h': 'H', 'd': 'D', 'c': 'C'}
    rank = rank_map[card[0]]
    suit = suit_map[card[1]]
    return suit + rank  # ✅ e.g., 'H5', 'DQ', 'CT'

def convert_to_treys(card):
    """
    Transforma (valoare, suit) în string în formatul 'As', 'Td', etc.
    Ex: (14, 's') -> 'As'
    """
    rank_map = {
        14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
         9: '9',  8: '8',  7: '7',  6: '6',  5: '5',
         4: '4',  3: '3',  2: '2'
    }
    rank = rank_map[card[0]]
    suit = card[1]  # 's', 'h', 'd', 'c'
    return rank + suit


def burn_card(deck):
    """Scoate o carte din pachet (o arde)."""
    if deck:
        deck.pop()

def evaluate_preflop_hand_strength(card1, card2, nb_simulation=200):
    """
    Estimate winrate of hand (card1, card2) vs random opponent using Monte Carlo simulation.
    """
    hole = [to_ppe(card1), to_ppe(card2)]
    win = 0

    for _ in range(nb_simulation):
        # Generate opponent hand (not overlapping)
        deck = [s + r for r in "23456789TJQKA" for s in "SHDC"]
        used = set(hole)
        deck = [c for c in deck if c not in used]
        opp_hand = random.sample(deck, 2)
        used.update(opp_hand)
        deck = [c for c in deck if c not in used]

        community = random.sample(deck, 5)

        ai_score = HandEvaluator.eval_hand(gen_cards(hole), gen_cards(community))
        opp_score = HandEvaluator.eval_hand(gen_cards(opp_hand), gen_cards(community))

        if ai_score > opp_score:
            continue  # lost
        elif ai_score < opp_score:
            win += 1
        else:
            win += 0.5  # tie

    return round(win / nb_simulation, 3)

def evaluate_hands(ai_hand, opponent_hand, board):
    evaluator = Evaluator()

    ai_cards = [Card.new(convert_to_treys(c)) for c in ai_hand]
    opp_cards = [Card.new(convert_to_treys(c)) for c in opponent_hand]
    board_cards = [Card.new(convert_to_treys(c)) for c in board]

    ai_score = evaluator.evaluate(board_cards, ai_cards)
    opp_score = evaluator.evaluate(board_cards, opp_cards)

    if ai_score < opp_score:
        return 1
    elif ai_score > opp_score:
        return -1
    else:
        return 0

def get_winrate_pypokerengine(ai_hand, board, nb_simulation=25):
    """
    Calculează probabilitatea de câștig a mâinii AI-ului în contextul actual.
    :param ai_hand: [(14, 'h'), (13, 'h')]  → 2 cărți proprii
    :param board: [(2, 'd'), (10, 'c')]     → 0–5 cărți comune
    """

    hole = [to_ppe(c) for c in ai_hand]
    community = [to_ppe(c) for c in board]

    win_rate = estimate_hole_card_win_rate(
        nb_simulation=nb_simulation,
        nb_player=2,
        hole_card=gen_cards(hole),
        community_card=gen_cards(community)
    )

    return round(win_rate, 3)


def get_hand_category(card1, card2):
    if card1[0] == card2[0]:
        return 0  # Pocket pair
    elif card1[1] == card2[1] and abs(card1[0] - card2[0]) == 1:
        return 1  # Suited connector
    elif card1[1] == card2[1]:
        return 2  # Suited
    elif abs(card1[0] - card2[0]) == 1:
        return 3  # Connector
    elif max(card1[0], card2[0]) >= 13:
        return 4  # High card (K or A)
    elif min(card1[0], card2[0]) <= 6:
        return 5  # Low cards
    else:
        return 6  # Other

def has_flush_draw(hand, board):
    suits = [c[1] for c in hand + board]
    for s in 'shdc':
        if suits.count(s) >= 4:
            return 1
    return 0

def has_straight_draw(hand, board):
    ranks = sorted(set([c[0] for c in hand + board]))
    for i in range(len(ranks) - 3):
        window = ranks[i:i+4]
        if window[-1] - window[0] <= 4:
            return 1
    return 0

def get_overcards_count(hand, board):
    if not board:
        return 0
    max_board = max([c[0] for c in board])
    return sum([1 for c in hand if c[0] > max_board])

def get_hand_strength(hand, board):
    from treys import Evaluator, Card
    evaluator = Evaluator()
    all_cards = hand + board
    treys_hand = [Card.new(convert_to_treys(c)) for c in hand]
    treys_board = [Card.new(convert_to_treys(c)) for c in board]
    score = evaluator.evaluate(treys_board, treys_hand)
    return score / 7462  # Normalize: 0 = best, 1 = worst

def get_hand_strength_bucket(score):
    if score < 0.2:
        return 4
    elif score < 0.4:
        return 3
    elif score < 0.6:
        return 2
    elif score < 0.8:
        return 1
    else:
        return 0

def get_aggression_factor(raise_count, call_count):
    return round(raise_count / (call_count + 1), 2)

def get_betting_pattern_index(last_action_ai1, last_action_ai2):
    return last_action_ai1 * 3 + last_action_ai2  # 0–8

def get_full_state(hand, board, position, pot_size, stack_ai, stack_opponent,
                   current_bet, round_stage, winrate,
                   last_action_ai1, last_action_ai2,
                   total_bet_opp, raises_this_street,
                   fold_count, call_count, raise_count,
                   total_hands):

    card1_val, card2_val = hand[0][0], hand[1][0]
    same_suit = int(hand[0][1] == hand[1][1])
    hand_cat = get_hand_category(hand[0], hand[1])
    preflop_wr = evaluate_preflop_hand_strength(hand[0], hand[1])

    street = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[round_stage]
    spr = min(stack_ai, stack_opponent) / (pot_size + 1e-6)

    flush_draw = has_flush_draw(hand, board)
    straight_draw = has_straight_draw(hand, board)
    overcards = get_overcards_count(hand, board)

    score = get_hand_strength(hand, board)
    bucket = get_hand_strength_bucket(score)

    aggression = get_aggression_factor(raise_count, call_count)
    pattern_idx = get_betting_pattern_index(last_action_ai1, last_action_ai2)

    total_hands = max(total_hands, 1)

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
        preflop_wr,
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
        aggression,
        pattern_idx
    ]

