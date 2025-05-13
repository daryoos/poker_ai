import random

suits = ['s', 'h', 'd', 'c']
ranks = list(range(2, 15))  # 2 - 14 (J=11, Q=12, ..., A=14)

def create_deck():
    return [(rank, suit) for rank in ranks for suit in suits]

def deal_hole_cards(deck):
    random.shuffle(deck)
    ai_hand = [deck.pop(), deck.pop()]
    opponent_hand = [deck.pop(), deck.pop()]
    return ai_hand, opponent_hand, deck

def burn_card(deck):
    if deck:
        deck.pop()

def convert_to_treys(card):
    rank_map = {14:'A', 13:'K', 12:'Q', 11:'J', 10:'T', 9:'9', 8:'8', 7:'7',
                6:'6', 5:'5', 4:'4', 3:'3', 2:'2'}
    return rank_map[card[0]] + card[1]
