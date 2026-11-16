"""A small, hand-defined PyTorch policy network for 5x5 Mini Quoridor."""
import torch
from torch import nn
from game import ACTION_COUNT, SIZE, GameState, legal_actions

INPUT_COUNT = SIZE * SIZE * 4 + 3  # 103 features


def encode_state(state: GameState) -> torch.Tensor:
    """Current player's viewpoint + walls + whose goal points upward."""
    features = [0.0] * INPUT_COUNT
    me, foe = state.turn, 1 - state.turn
    for channel, player in enumerate((me, foe)):
        r, c = state.pawns[player]
        features[channel * 25 + r * SIZE + c] = 1.0
    for orientation, r, c in state.walls:
        channel = 2 if orientation == 'h' else 3
        features[channel * 25 + r * SIZE + c] = 1.0
    features[100] = state.remaining[me] / 3.0
    features[101] = state.remaining[foe] / 3.0
    features[102] = float(me)  # Goal direction indicator
    return torch.tensor(features, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_COUNT, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def masked_logits(model: PolicyNetwork, state: GameState) -> torch.Tensor:
    logits = model(encode_state(state))
    legal = legal_actions(state)
    if not legal:
        raise ValueError('Game already finished')
    mask = torch.full_like(logits, -1e9)
    mask[legal] = 0.0
    return logits + mask


def choose_neural_action(model: PolicyNetwork, state: GameState) -> int:
    with torch.no_grad():
        return int(masked_logits(model, state).argmax().item())


def load_model(path: str, device: str = 'cpu') -> PolicyNetwork:
    model = PolicyNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
