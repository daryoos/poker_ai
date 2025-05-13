from treys import Evaluator, Card
from cards import convert_to_treys

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
