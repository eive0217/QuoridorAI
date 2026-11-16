"""Experimental self-play REINFORCE training (not a guaranteed strong AI).

Usage:
    python train.py --episodes 3000
    python train.py --episodes 10000 --save models/mini_quoridor.pth
"""
import argparse
import os
import random
from collections import deque

import torch
from torch.distributions import Categorical

from game import GameState, apply_action, shortest_distance
from neural_net import PolicyNetwork, masked_logits
from agents import heuristic_action


def train(episodes: int, save_path: str, learning_rate: float, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    model = PolicyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recent_wins = deque(maxlen=100)
    max_turns = 150  # Draw cap to prevent looping forever
    model.train()

    for episode in range(1, episodes + 1):
        state = GameState()
        trajectories = {0: [], 1: []}
        # Randomize player sides by occasional reflection of opening positions
        if random.random() < 0.5:
            state.pawns = [(4, random.randrange(5)), (0, random.randrange(5))]
        for _ in range(max_turns):
            who = state.turn
            before = shortest_distance(state, who)
            # Mix in heuristic opponent sometimes, for less sparse practice.
            heuristic_turn = (episode % 4 == 0 and who == 1)
            if heuristic_turn:
                action = heuristic_action(state)
                logprob, entropy = None, None
            else:
                distribution = Categorical(logits=masked_logits(model, state))
                selection = distribution.sample()
                action = int(selection.item())
                logprob = distribution.log_prob(selection)
                entropy = distribution.entropy()
            next_state = apply_action(state, action)
            after = shortest_distance(next_state, who)
            # Dense route-progress shaping plus terminal win/loss reward.
            reward = 0.03 * max(-3, min(3, before - after)) - 0.002
            if next_state.winner is not None:
                reward += 1.0
            trajectories[who].append((logprob, entropy, reward))
            state = next_state
            if state.winner is not None:
                break

        winner = state.winner
        recent_wins.append(-1 if winner is None else winner)
        losses = []
        for player, sequence in trajectories.items():
            if not sequence:
                continue
            rewards = [x[2] for x in sequence]
            if winner is not None and winner != player:
                rewards[-1] -= 1.0
            # Discounted rewards across this player's own actions.
            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + 0.98 * running
                returns.append(running)
            returns.reverse()
            for (logprob, entropy, _), ret in zip(sequence, returns):
                if logprob is not None:
                    losses.append((-logprob * ret) - 0.01 * entropy)
        if losses:
            loss = torch.stack(losses).sum() / len(losses)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if episode % 100 == 0 or episode == 1:
            wins0 = sum(w == 0 for w in recent_wins)
            wins1 = sum(w == 1 for w in recent_wins)
            draws = sum(w == -1 for w in recent_wins)
            print(f'Episode {episode}/{episodes} | last {len(recent_wins)}: '
                  f'P0 {wins0}, P1 {wins1}, draws {draws}')
        if episode % 1000 == 0:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Saved neural-network weights: {save_path}')
    print('Note: this is an experimental baseline. Evaluate strength against bots before claiming improvement.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--save', type=str, default='models/mini_quoridor.pth')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.episodes, args.save, args.lr, args.seed)
