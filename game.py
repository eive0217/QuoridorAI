"""Pure-Python Mini Quoridor rules. No pygame or torch dependencies."""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

SIZE = 5
WALLS_PER_PLAYER = 3
# Action indices: 0..24: destination square (row*5+col)
# 25..40: horizontal wall anchors; 41..56: vertical wall anchors.
ACTION_COUNT = SIZE * SIZE + 2 * (SIZE - 1) ** 2


@dataclass
class GameState:
    pawns: list[tuple[int, int]] = field(default_factory=lambda: [(4, 2), (0, 2)])
    # Wall entries: (orientation, row, column), orientation 'h' or 'v'.
    walls: set[tuple[str, int, int]] = field(default_factory=set)
    remaining: list[int] = field(default_factory=lambda: [WALLS_PER_PLAYER] * 2)
    turn: int = 0
    winner: Optional[int] = None
    moves_played: int = 0

    def clone(self) -> 'GameState':
        return GameState(list(self.pawns), set(self.walls), list(self.remaining),
                         self.turn, self.winner, self.moves_played)


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < SIZE and 0 <= c < SIZE


def neighbors(square: tuple[int, int]):
    r, c = square
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nxt = (r + dr, c + dc)
        if in_bounds(*nxt):
            yield nxt


def is_blocked(state: GameState, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether an existing wall separates two adjacent squares."""
    ar, ac = a
    br, bc = b
    if abs(ar - br) + abs(ac - bc) != 1:
        raise ValueError('is_blocked requires adjacent squares')
    if ar != br:
        top = min(ar, br)
        return any(('h', top, k) in state.walls for k in (ac - 1, ac) if 0 <= k < SIZE - 1)
    left = min(ac, bc)
    return any(('v', k, left) in state.walls for k in (ar - 1, ar) if 0 <= k < SIZE - 1)


def shortest_distance(state: GameState, player: int) -> int:
    """Shortest route to goal, ignoring pawns but respecting walls."""
    start = state.pawns[player]
    goal_row = 0 if player == 0 else SIZE - 1
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        square, distance = queue.popleft()
        if square[0] == goal_row:
            return distance
        for nxt in neighbors(square):
            if nxt not in visited and not is_blocked(state, square, nxt):
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    return 10**6


def legal_pawn_moves(state: GameState) -> list[tuple[int, int]]:
    """Quoridor movement: orthogonal step, straight jump, or side-step around blocked jump."""
    if state.winner is not None:
        return []
    here = state.pawns[state.turn]
    foe = state.pawns[1 - state.turn]
    result = set()
    for adj in neighbors(here):
        if is_blocked(state, here, adj):
            continue
        if adj != foe:
            result.add(adj)
            continue
        dr, dc = adj[0] - here[0], adj[1] - here[1]
        behind = (foe[0] + dr, foe[1] + dc)
        if in_bounds(*behind) and not is_blocked(state, foe, behind):
            result.add(behind)
        else:
            # Opponent blocks the adjacent square; if straight jump isn't
            # possible, diagonal moves around that pawn become legal.
            for side in neighbors(foe):
                if side == here or side == behind:
                    continue
                if not is_blocked(state, foe, side):
                    result.add(side)
    return sorted(result)


def can_place_wall(state: GameState, orientation: str, r: int, c: int) -> bool:
    if (state.winner is not None or state.remaining[state.turn] <= 0 or
            orientation not in ('h', 'v') or not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1)):
        return False
    wall = (orientation, r, c)
    if wall in state.walls or (('v' if orientation == 'h' else 'h'), r, c) in state.walls:
        return False  # collision or perpendicular crossing at midpoint
    # Two collinear walls cannot overlap one half-section.
    if orientation == 'h' and any(('h', r, k) in state.walls for k in (c - 1, c + 1)):
        return False
    if orientation == 'v' and any(('v', k, c) in state.walls for k in (r - 1, r + 1)):
        return False
    candidate = state.clone()
    candidate.walls.add(wall)
    return all(shortest_distance(candidate, player) < 10**6 for player in (0, 1))


def wall_action(orientation: str, r: int, c: int) -> int:
    if orientation not in ('h', 'v') or not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1):
        raise ValueError('Bad wall position')
    return SIZE * SIZE + (0 if orientation == 'h' else (SIZE - 1) ** 2) + r * (SIZE - 1) + c


def decode_wall(action: int) -> tuple[str, int, int]:
    if not SIZE * SIZE <= action < ACTION_COUNT:
        raise ValueError('Not a wall action')
    index = action - SIZE * SIZE
    orientation = 'h' if index < (SIZE - 1) ** 2 else 'v'
    index %= (SIZE - 1) ** 2
    return orientation, index // (SIZE - 1), index % (SIZE - 1)


def legal_actions(state: GameState) -> list[int]:
    if state.winner is not None:
        return []
    moves = [r * SIZE + c for r, c in legal_pawn_moves(state)]
    if state.remaining[state.turn]:
        moves.extend(
            wall_action(o, r, c)
            for o in ('h', 'v')
            for r in range(SIZE - 1)
            for c in range(SIZE - 1)
            if can_place_wall(state, o, r, c)
        )
    return moves


def apply_action(state: GameState, action: int) -> GameState:
    """Return a new state; reject any illegal move. Never mutate input."""
    if action not in legal_actions(state):
        raise ValueError(f'Illegal action: {action}')
    new = state.clone()
    actor = new.turn
    if action < SIZE * SIZE:
        square = divmod(action, SIZE)
        new.pawns[actor] = square
        if square[0] == (0 if actor == 0 else SIZE - 1):
            new.winner = actor
    else:
        new.walls.add(decode_wall(action))
        new.remaining[actor] -= 1
    new.moves_played += 1
    new.turn = 1 - actor
    return new
