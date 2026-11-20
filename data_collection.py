"""Record completed human-vs-heuristic games as JSONL, without PyTorch.

State format v1 (84 floats):
  0..24   current player's pawn, row-major 5x5
 25..49   opponent pawn, row-major 5x5
 50..65   horizontal wall anchors, row-major 4x4
 66..81   vertical wall anchors, row-major 4x4
 82..83   remaining wall ratios (self, opponent)

For player 1 the board is reflected vertically: both players always aim
UP in the encoded state. Actions are reflected using the same convention.
"""
import json
from pathlib import Path

from game import SIZE, WALLS_PER_PLAYER, GameState, legal_actions, decode_wall, wall_action

INPUT_COUNT = 84
SCHEMA_VERSION = 1
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "human_games.jsonl"


def canonical_action(action: int, player: int) -> int:
    """Turn physical board action into current-player-up action index."""
    if player == 0:
        return action
    if action < SIZE * SIZE:
        row, col = divmod(action, SIZE)
        return (SIZE - 1 - row) * SIZE + col
    direction, row, col = decode_wall(action)
    return wall_action(direction, SIZE - 2 - row, col)


def encode_state_84(state: GameState) -> list[float]:
    """Encode the position from the current player's perspective."""
    me = state.turn
    result = [0.0] * INPUT_COUNT

    for idx, player in enumerate((me, 1 - me)):
        row, col = state.pawns[player]
        if me == 1:
            row = SIZE - 1 - row
        result[idx * 25 + row * SIZE + col] = 1.0

    for direction, row, col in state.walls:
        if me == 1:
            row = SIZE - 2 - row
        offset = 50 if direction == "h" else 66
        result[offset + row * (SIZE - 1) + col] = 1.0

    result[82] = state.remaining[me] / WALLS_PER_PLAYER
    result[83] = state.remaining[1 - me] / WALLS_PER_PLAYER
    return result


def completed_count(path=DEFAULT_PATH) -> int:
    """Count valid completed heuristic games already saved."""
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("opponent") == "heuristic" and obj.get("result") in ("win", "loss"):
                count += 1
    return count


class HumanGameRecorder:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.moves = []
        self.game_id = completed_count(self.path) + 1

    def discard(self):
        """Discard unfinished game on reset or exit."""
        self.moves = []

    def record(self, state: GameState, action: int):
        """Call before apply_action, for the human player's moves only."""
        if state.turn != 0 or action not in legal_actions(state):
            raise ValueError("Can only record a legal human action before it is played")
        self.moves.append({
            "turn": len(self.moves) + 1,
            "state": encode_state_84(state),
            "action": canonical_action(action, state.turn),
            "legal_actions": sorted(canonical_action(a, state.turn) for a in legal_actions(state)),
        })

    def finish(self, state: GameState) -> bool:
        """Append exactly one completed game, then clear pending moves."""
        if state.winner is None or not self.moves:
            return False
        record = {
            "schema_version": SCHEMA_VERSION,
            "game_id": self.game_id,
            "opponent": "heuristic",
            "result": "win" if state.winner == 0 else "loss",
            "moves": self.moves,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.moves = []
        self.game_id += 1
        return True
