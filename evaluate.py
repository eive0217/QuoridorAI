"""Measure the trained model against random and shortest-path opponents."""
import argparse
import random
from game import GameState, apply_action
from agents import random_action, heuristic_action
from neural_net import load_model, choose_neural_action


def evaluate(path: str, games: int, opponent: str):
    random.seed(123)
    model = load_model(path)
    bot = random_action if opponent == 'random' else heuristic_action
    results = {'win': 0, 'loss': 0, 'draw': 0}
    for i in range(games):
        state = GameState()
        neural_player = i % 2  # alternate who goes first
        for _ in range(150):
            action = choose_neural_action(model, state) if state.turn == neural_player else bot(state)
            state = apply_action(state, action)
            if state.winner is not None:
                break
        outcome = 'draw' if state.winner is None else ('win' if state.winner == neural_player else 'loss')
        results[outcome] += 1
    print(f'{games} games vs {opponent}: {results} | win rate = {100 * results["win"] / games:.1f}%')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/mini_quoridor.pth')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--opponent', choices=['random', 'heuristic'], default='random')
    args = p.parse_args()
    evaluate(args.model, args.games, args.opponent)
