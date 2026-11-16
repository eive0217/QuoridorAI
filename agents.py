"""Simple opponents for testing a neural-network agent."""
import random
from game import GameState, apply_action, legal_actions, shortest_distance, SIZE


def random_action(state: GameState) -> int:
    return random.choice(legal_actions(state))


def heuristic_action(state: GameState) -> int:
    """Short-path baseline; occasionally blocks the other player."""
    actions = legal_actions(state)
    actor = state.turn
    best_score = float('-inf')
    best = []
    for action in actions:
        next_state = apply_action(state, action)
        if next_state.winner == actor:
            return action
        mine = shortest_distance(next_state, actor)
        theirs = shortest_distance(next_state, 1 - actor)
        # Slight wall penalty: don't block unless it helps enough.
        score = 1.1 * theirs - mine - (0.7 if action >= SIZE * SIZE else 0.0)
        score += random.random() * 0.03
        if score > best_score:
            best_score, best = score, [action]
        elif score == best_score:
            best.append(action)
    return random.choice(best)
