"""Playable Pygame front-end. Start with `python main.py`."""
import os
import sys
import pygame

from game import (GameState, SIZE, apply_action, legal_pawn_moves,
                  can_place_wall, wall_action)
from agents import heuristic_action, random_action

WIDTH, HEIGHT = 650, 675
X0, Y0 = 105, 142
STEP, TILE = 88, 74
BG = (20, 30, 45)
BOARD_BG = (74, 60, 47)
TILE_1, TILE_2 = (215, 194, 160), (202, 176, 138)
BLUE, RED = (62, 148, 240), (242, 103, 98)
WHITE, MUTED = (235, 241, 247), (168, 179, 196)
WALL = (64, 43, 27)
HIGHLIGHT = (52, 171, 132)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mini_quoridor.pth')


def tile_rect(r, c):
    return pygame.Rect(X0 + c * STEP, Y0 + r * STEP, TILE, TILE)


def wall_rect(o, r, c):
    if o == 'h':
        return pygame.Rect(X0 + c * STEP, Y0 + r * STEP + TILE + 2, TILE + STEP, 10)
    return pygame.Rect(X0 + c * STEP + TILE + 2, Y0 + r * STEP, 10, TILE + STEP)


def nearest_anchor(pos):
    x, y = pos
    choices = []
    for r in range(4):
        for c in range(4):
            mid_x = X0 + c * STEP + TILE + (STEP - TILE) // 2
            mid_y = Y0 + r * STEP + TILE + (STEP - TILE) // 2
            dist2 = (x - mid_x) ** 2 + (y - mid_y) ** 2
            choices.append((dist2, r, c))
    dist, r, c = min(choices)
    return (r, c) if dist <= 42**2 else None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Mini Quoridor - Human vs AI')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 22)
    small = pygame.font.SysFont('arial', 17)
    big = pygame.font.SysFont('arial', 30, bold=True)
    state = GameState()
    wall_mode = False
    orientation = 'h'
    ai_kind = 'heuristic'
    neural = None
    status = 'Move: click a green-highlighted tile.'

    def text(message, x, y, color=WHITE, face=None):
        screen.blit((face or font).render(message, True, color), (x, y))

    def reset():
        nonlocal state, wall_mode, status
        state = GameState()
        wall_mode = False
        status = 'New game! Blue moves first.'

    def select_neural():
        nonlocal ai_kind, neural, status
        if not os.path.exists(MODEL_PATH):
            status = 'No trained model yet. Run: python train.py'
            return
        try:
            from neural_net import load_model
            neural = load_model(MODEL_PATH)
            ai_kind = 'neural'
            reset()
            status = 'Neural AI loaded. Play!'
        except Exception as exc:
            status = f'Model loading failed: {str(exc)[:52]}'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_m:
                    wall_mode = not wall_mode
                elif event.key == pygame.K_h:
                    orientation = 'h'
                    wall_mode = True
                elif event.key == pygame.K_v:
                    orientation = 'v'
                    wall_mode = True
                elif event.key == pygame.K_1:
                    ai_kind = 'random'
                    reset()
                elif event.key == pygame.K_2:
                    ai_kind = 'heuristic'
                    reset()
                elif event.key == pygame.K_3:
                    select_neural()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state.turn != 0 or state.winner is not None:
                    continue
                chosen = None
                if wall_mode:
                    anchor = nearest_anchor(event.pos)
                    if anchor and can_place_wall(state, orientation, *anchor):
                        chosen = wall_action(orientation, *anchor)
                else:
                    for r, c in legal_pawn_moves(state):
                        if tile_rect(r, c).collidepoint(event.pos):
                            chosen = r * SIZE + c
                            break
                if chosen is not None:
                    state = apply_action(state, chosen)
                    if state.winner is None:
                        status = 'AI is thinking...'

        if state.turn == 1 and state.winner is None:
            if ai_kind == 'random':
                action = random_action(state)
            elif ai_kind == 'neural' and neural is not None:
                from neural_net import choose_neural_action
                action = choose_neural_action(neural, state)
            else:
                action = heuristic_action(state)
            state = apply_action(state, action)
            status = 'Your turn. Press M to toggle wall mode.'

        screen.fill(BG)
        text('MINI QUORIDOR', 105, 28, WHITE, big)
        text('Blue: YOU     Red: AI', 106, 73, MUTED, small)
        text(f'AI: {ai_kind.upper()}     |     Blue walls: {state.remaining[0]}     Red walls: {state.remaining[1]}',
             105, 100, WHITE, small)

        pygame.draw.rect(screen, BOARD_BG, (X0 - 14, Y0 - 14, 5 * STEP + 2, 5 * STEP + 2), border_radius=10)
        for r in range(SIZE):
            for c in range(SIZE):
                rect = tile_rect(r, c)
                pygame.draw.rect(screen, TILE_1 if (r + c) % 2 == 0 else TILE_2,
                                 rect, border_radius=7)
        if state.turn == 0 and state.winner is None and not wall_mode:
            for r, c in legal_pawn_moves(state):
                pygame.draw.circle(screen, HIGHLIGHT, tile_rect(r, c).center, 10)
        for o, r, c in state.walls:
            pygame.draw.rect(screen, WALL, wall_rect(o, r, c), border_radius=5)

        if wall_mode and state.turn == 0 and state.winner is None:
            anchor = nearest_anchor(pygame.mouse.get_pos())
            if anchor:
                o, (r, c) = orientation, anchor
                preview = pygame.Surface(wall_rect(o, r, c).size, pygame.SRCALPHA)
                ok = can_place_wall(state, o, r, c)
                preview.fill((*HIGHLIGHT, 200) if ok else (*RED, 170))
                screen.blit(preview, wall_rect(o, r, c).topleft)
        for player, color in ((0, BLUE), (1, RED)):
            center = tile_rect(*state.pawns[player]).center
            pygame.draw.circle(screen, (23, 29, 38), (center[0] + 3, center[1] + 5), 25)
            pygame.draw.circle(screen, color, center, 23)
            pygame.draw.circle(screen, WHITE, center, 23, width=3)

        if state.winner is not None:
            text('YOU WIN!' if state.winner == 0 else 'AI WINS!', 230, 600, HIGHLIGHT if state.winner == 0 else RED, big)
        else:
            text('MODE: WALL' if wall_mode else 'MODE: MOVE', 105, 602, HIGHLIGHT if wall_mode else WHITE, font)
        text(status[:65], 105, 638, MUTED, small)
        text('M: Mode | H/V: Wall direction | R: Reset', 105, 569, WHITE, small)
        text('1: Random AI | 2: Heuristic AI | 3: Neural AI', 105, 587, WHITE, small)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
