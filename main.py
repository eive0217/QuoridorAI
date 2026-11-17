"""Clean, high-contrast Mini Quoridor interface. Run: python main.py."""
import os
import sys
import pygame

from game import GameState, SIZE, apply_action, legal_pawn_moves, can_place_wall, wall_action
from agents import heuristic_action, random_action

WIDTH, HEIGHT = 720, 900
CELL, GAP = 65, 13
STEP = CELL + GAP
BOARD_SIZE = SIZE * CELL + (SIZE - 1) * GAP
X0 = (WIDTH - BOARD_SIZE) // 2
Y0 = 339

WHITE = (255, 255, 255)
BLACK = (20, 26, 23)
GREEN = (17, 112, 77)
GREEN_DARK = (11, 82, 56)
GREEN_SOFT = (218, 240, 228)
PALE = (245, 249, 246)
LINE = (219, 229, 223)
MUTED = (91, 106, 98)
BLUE = (47, 117, 230)
RED = (224, 75, 80)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mini_quoridor.pth')


def tile_rect(r, c):
    return pygame.Rect(X0 + c * STEP, Y0 + r * STEP, CELL, CELL)


def wall_rect(orientation, r, c):
    """Draw thick, visible walls centered inside the board's tile gaps."""
    thickness = 12
    if orientation == 'h':
        return pygame.Rect(X0 + c * STEP, Y0 + r * STEP + CELL + (GAP-thickness)//2,
                           CELL * 2 + GAP, thickness)
    return pygame.Rect(X0 + c * STEP + CELL + (GAP-thickness)//2, Y0 + r * STEP,
                       thickness, CELL * 2 + GAP)


def nearest_anchor(pos):
    x, y = pos
    choices = []
    for r in range(SIZE - 1):
        for c in range(SIZE - 1):
            center_x = X0 + c * STEP + CELL + GAP // 2
            center_y = Y0 + r * STEP + CELL + GAP // 2
            distance = (x - center_x) ** 2 + (y - center_y) ** 2
            choices.append((distance, r, c))
    distance, r, c = min(choices)
    return (r, c) if distance <= 47 ** 2 else None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('MINI QUORIDOR | You vs AI')
    clock = pygame.time.Clock()
    # Arial is the only family requested for every piece of UI text.
    font = pygame.font.SysFont('arial', 20)
    small = pygame.font.SysFont('arial', 16)
    tiny = pygame.font.SysFont('arial', 14)
    mid = pygame.font.SysFont('arial', 22, bold=True)
    title = pygame.font.SysFont('arial', 34, bold=True)

    state = GameState()
    wall_mode = False
    orientation = 'h'
    ai_kind = 'heuristic'
    neural = None
    status = 'Click a highlighted square to move.'

    mode_move = pygame.Rect(175, 790, 115, 40)
    mode_h = pygame.Rect(302, 790, 115, 40)
    mode_v = pygame.Rect(429, 790, 115, 40)

    def draw_text(message, x, y, color=BLACK, face=None):
        screen.blit((face or font).render(str(message), True, color), (x, y))

    def draw_panel(rect, color=WHITE, border=LINE, radius=14):
        pygame.draw.rect(screen, color, rect, border_radius=radius)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)

    def draw_pill(rect, label, selected):
        pygame.draw.rect(screen, GREEN if selected else WHITE, rect, border_radius=9)
        pygame.draw.rect(screen, GREEN if selected else LINE, rect, width=1, border_radius=9)
        rendered = small.render(label, True, WHITE if selected else BLACK)
        screen.blit(rendered, rendered.get_rect(center=rect.center))

    def reset():
        nonlocal state, wall_mode, status
        state = GameState()
        wall_mode = False
        status = 'New game. Your turn!'

    def select_neural():
        nonlocal ai_kind, neural, status
        if not os.path.exists(MODEL_PATH):
            status = 'No trained model. First run: python train.py'
            return
        try:
            from neural_net import load_model
            neural = load_model(MODEL_PATH)
            ai_kind = 'neural'
            reset()
            status = 'Neural-network AI loaded.'
        except Exception as exc:
            status = f'Model loading failed: {str(exc)[:60]}'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_m:
                    wall_mode = not wall_mode
                elif event.key == pygame.K_h:
                    orientation, wall_mode = 'h', True
                elif event.key == pygame.K_v:
                    orientation, wall_mode = 'v', True
                elif event.key == pygame.K_1:
                    ai_kind = 'random'
                    reset()
                elif event.key == pygame.K_2:
                    ai_kind = 'heuristic'
                    reset()
                elif event.key == pygame.K_3:
                    select_neural()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mode_move.collidepoint(event.pos):
                    wall_mode = False
                    continue
                if mode_h.collidepoint(event.pos):
                    wall_mode, orientation = True, 'h'
                    continue
                if mode_v.collidepoint(event.pos):
                    wall_mode, orientation = True, 'v'
                    continue
                if state.turn != 0 or state.winner is not None:
                    continue
                chosen = None
                if wall_mode:
                    anchor = nearest_anchor(event.pos)
                    if anchor and can_place_wall(state, orientation, *anchor):
                        chosen = wall_action(orientation, *anchor)
                    elif anchor:
                        status = 'Cannot place a wall here. Try another gap.'
                else:
                    for r, c in legal_pawn_moves(state):
                        if tile_rect(r, c).collidepoint(event.pos):
                            chosen = r * SIZE + c
                            break
                if chosen is not None:
                    state = apply_action(state, chosen)
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
            status = 'Your turn. Move or place a wall.'

        screen.fill(WHITE)
        # All gameplay instructions at the TOP.
        draw_text('MINI QUORIDOR', 44, 24, BLACK, title)
        draw_text('HOW TO PLAY', 44, 81, GREEN_DARK, small)
        draw_text('Reach the opposite edge first. Click a highlighted tile to move.', 44, 106, BLACK, small)
        draw_text('Place walls across two spaces to block paths. Never block all routes.', 44, 130, BLACK, small)
        draw_text('Controls: M = mode  |  H/V = wall direction  |  R = restart', 44, 154, MUTED, small)
        draw_text('AI difficulty: 1 = random  |  2 = heuristic  |  3 = neural network', 44, 176, MUTED, small)
        pygame.draw.line(screen, LINE, (44, 209), (676, 209), 1)

        # AI information is located ABOVE the board.
        ai_panel = pygame.Rect(154, 222, 412, 90)
        draw_panel(ai_panel, PALE)
        pygame.draw.circle(screen, RED, (187, 265), 15)
        pygame.draw.circle(screen, WHITE, (187, 265), 15, 2)
        draw_text('AI  /  RED', 215, 238, BLACK, mid)
        draw_text(f'{ai_kind.upper()}  |  Walls left: {state.remaining[1]}', 215, 270, MUTED, small)

        # Board has NO brown background or wooden checkerboard.
        frame = pygame.Rect(X0-16, Y0-16, BOARD_SIZE+32, BOARD_SIZE+32)
        draw_panel(frame, WHITE, LINE, 14)
        for r in range(SIZE):
            for c in range(SIZE):
                pygame.draw.rect(screen, GREEN_SOFT, tile_rect(r, c), border_radius=7)
                pygame.draw.rect(screen, LINE, tile_rect(r, c), 1, border_radius=7)

        if state.turn == 0 and state.winner is None and not wall_mode:
            for r, c in legal_pawn_moves(state):
                center = tile_rect(r, c).center
                pygame.draw.circle(screen, GREEN, center, 10)

        # Very visible solid green walls drawn ON TOP of tiles.
        for o, r, c in sorted(state.walls):
            rect = wall_rect(o, r, c)
            pygame.draw.rect(screen, GREEN_DARK, rect, border_radius=5)
            pygame.draw.rect(screen, GREEN, rect.inflate(-3, -3), border_radius=4)

        if wall_mode and state.turn == 0 and state.winner is None:
            anchor = nearest_anchor(pygame.mouse.get_pos())
            if anchor:
                r, c = anchor
                rect = wall_rect(orientation, r, c)
                ok = can_place_wall(state, orientation, r, c)
                color = GREEN if ok else RED
                ghost = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(ghost, (*color, 170), ghost.get_rect(), border_radius=5)
                screen.blit(ghost, rect.topleft)

        for player, color in ((0, BLUE), (1, RED)):
            center = tile_rect(*state.pawns[player]).center
            pygame.draw.circle(screen, WHITE, center, 24)
            pygame.draw.circle(screen, color, center, 21)
            pygame.draw.circle(screen, WHITE, center, 21, 3)

        # Player information is BELOW the board.
        player_panel = pygame.Rect(154, 734, 412, 86)
        draw_panel(player_panel, PALE)
        pygame.draw.circle(screen, BLUE, (187, 774), 15)
        pygame.draw.circle(screen, WHITE, (187, 774), 15, 2)
        draw_text('YOU  /  BLUE', 215, 748, BLACK, mid)
        draw_text(f'Walls left: {state.remaining[0]}', 215, 779, MUTED, small)

        # Buttons sit under player details to keep the board unobstructed.
        # Button positions are placed below the player information panel.
        mode_move.y = mode_h.y = mode_v.y = 831
        draw_pill(mode_move, 'MOVE', not wall_mode)
        draw_pill(mode_h, 'WALL  H', wall_mode and orientation == 'h')
        draw_pill(mode_v, 'WALL  V', wall_mode and orientation == 'v')

        if state.winner is not None:
            message = 'YOU WIN!' if state.winner == 0 else 'AI WINS!'
            message_color = GREEN_DARK if state.winner == 0 else RED
        else:
            message = status
            message_color = GREEN_DARK
        rendered = tiny.render(message, True, message_color)
        screen.blit(rendered, rendered.get_rect(center=(WIDTH//2, 884)))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
