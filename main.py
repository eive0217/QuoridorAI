
"""600x600 Mini Quoridor - Black UI."""

import os
import sys
import pygame

from game import (
    GameState, SIZE, apply_action,
    legal_pawn_moves, can_place_wall, wall_action
)
from agents import heuristic_action, random_action
from data_collection import HumanGameRecorder, completed_count


# --------------------
# WINDOW
# --------------------

WIDTH = HEIGHT = 600
CELL = 56
GAP = 8
STEP = CELL + GAP

BOARD = CELL * SIZE + GAP * (SIZE - 1)
X0 = (WIDTH - BOARD) // 2
Y0 = 149


# --------------------
# COLORS
# --------------------

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (44, 210, 119)
DIM = (173, 173, 173)
BLUE = (53, 128, 255)
RED = (245, 77, 85)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "mini_quoridor.pth"
)


# --------------------
# BUTTONS
# --------------------

BUTTONS = {
    "move": pygame.Rect(123, 524, 110, 30),
    "h": pygame.Rect(245, 524, 110, 30),
    "v": pygame.Rect(367, 524, 110, 30),
}


# --------------------
# BOARD GEOMETRY
# --------------------

def tile_rect(row, col):
    return pygame.Rect(
        X0 + col * STEP,
        Y0 + row * STEP,
        CELL,
        CELL
    )


def wall_rect(orientation, row, col):

    if orientation == "h":
        return pygame.Rect(
            X0 + col * STEP,
            Y0 + row * STEP + CELL,
            2 * CELL + GAP,
            GAP
        )

    return pygame.Rect(
        X0 + col * STEP + CELL,
        Y0 + row * STEP,
        GAP,
        2 * CELL + GAP
    )


def nearest_anchor(pos):

    x, y = pos
    candidates = []

    for row in range(SIZE - 1):
        for col in range(SIZE - 1):

            cx = X0 + col * STEP + CELL + GAP / 2
            cy = Y0 + row * STEP + CELL + GAP / 2

            distance = (
                (x - cx) ** 2 +
                (y - cy) ** 2
            )

            candidates.append(
                (distance, row, col)
            )

    distance, row, col = min(candidates)

    if distance <= 43 ** 2:
        return row, col

    return None


# --------------------
# MAIN
# --------------------

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "MINI QUORIDOR"
    )

    clock = pygame.time.Clock()

    # Arial font with antialiasing
    arial = pygame.font.match_font("arial")

    if not arial:
        print("WARNING: Arial is not installed.")

    def font(size, bold=False):
        if bold:
            return pygame.font.SysFont(
                "arial", size, bold=True
            )
        return pygame.font.Font(arial, size)

    title_font = font(25, True)
    label_font = font(17, True)
    body_font = font(15)
    info_font = font(14)
    tiny_font = font(13)

    state = GameState()
    recorder = HumanGameRecorder()
    DATA_TARGET = 30

    wall_mode = False
    orientation = "h"

    ai_kind = "heuristic"
    neural = None

    status = "YOUR TURN - SELECT A MOVE OR WALL"

    # --------------------
    # TEXT
    # --------------------

    def text(
        message,
        x,
        y,
        face=body_font,
        color=WHITE,
        center=False
    ):

        surface = face.render(
            str(message),
            True,
            color
        )

        rect = surface.get_rect()

        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)

        screen.blit(surface, rect)

    # --------------------
    # RESET
    # --------------------

    def reset():

        nonlocal state, wall_mode, status

        recorder.discard()  # R/AI switch discards unfinished games
        state = GameState()
        wall_mode = False

        status = "YOUR TURN - SELECT A MOVE OR WALL"

    # --------------------
    # LOAD NEURAL AI
    # --------------------

    def load_neural():

        nonlocal ai_kind, neural, status

        if not os.path.exists(MODEL_PATH):
            status = "NO MODEL FOUND - RUN TRAIN.PY FIRST"
            return

        try:
            from neural_net import load_model

            neural = load_model(MODEL_PATH)
            ai_kind = "neural"

            reset()

        except Exception as exc:
            status = (
                "COULD NOT LOAD MODEL: "
                + str(exc)[:40]
            )

    # --------------------
    # GAME LOOP
    # --------------------

    running = True

    while running:

        # --------------------
        # EVENTS
        # --------------------

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
                    wall_mode = True
                    orientation = "h"

                elif event.key == pygame.K_v:
                    wall_mode = True
                    orientation = "v"

                elif event.key == pygame.K_1:
                    ai_kind = "random"
                    reset()

                elif event.key == pygame.K_2:
                    ai_kind = "heuristic"
                    reset()

                elif event.key == pygame.K_3:
                    load_neural()

            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):

                if BUTTONS["move"].collidepoint(event.pos):
                    wall_mode = False
                    continue

                if BUTTONS["h"].collidepoint(event.pos):
                    wall_mode = True
                    orientation = "h"
                    continue

                if BUTTONS["v"].collidepoint(event.pos):
                    wall_mode = True
                    orientation = "v"
                    continue

                if (
                    state.turn != 0
                    or state.winner is not None
                ):
                    continue

                action = None

                if wall_mode:

                    anchor = nearest_anchor(event.pos)

                    if anchor is not None:

                        if can_place_wall(
                            state,
                            orientation,
                            *anchor
                        ):
                            action = wall_action(
                                orientation,
                                *anchor
                            )

                        else:
                            status = "INVALID WALL POSITION"

                else:

                    for row, col in legal_pawn_moves(state):

                        if tile_rect(
                            row, col
                        ).collidepoint(event.pos):

                            action = row * SIZE + col
                            break

                if action is not None:
                    if ai_kind == "heuristic" and completed_count() < DATA_TARGET:
                        recorder.record(state, action)  # BEFORE the state changes
                    state = apply_action(state, action)

        # --------------------
        # AI TURN
        # --------------------

        if (
            state.turn == 1
            and state.winner is None
        ):

            if ai_kind == "random":
                action = random_action(state)

            elif (
                ai_kind == "neural"
                and neural is not None
            ):

                from neural_net import choose_neural_action

                action = choose_neural_action(
                    neural,
                    state
                )

            else:
                action = heuristic_action(state)

            state = apply_action(state, action)

            status = "YOUR TURN - SELECT A MOVE OR WALL"

        # Save each completed human-vs-heuristic game exactly once.
        # Unfinished games, random games, and neural games are excluded.
        if state.winner is not None and ai_kind == "heuristic":
            recorder.finish(state)

        # --------------------
        # BACKGROUND
        # --------------------

        screen.fill(BLACK)

        # --------------------
        # GAME INSTRUCTIONS
        # --------------------

        text(
            "MINI QUORIDOR",
            WIDTH // 2,
            22,
            title_font,
            WHITE,
            True
        )

        text(
            "Reach the opposite edge first. Click a green dot to move.",
            WIDTH // 2,
            49,
            info_font,
            WHITE,
            True
        )

        text(
            "Walls block paths, but cannot seal off every route.",
            WIDTH // 2,
            67,
            info_font,
            WHITE,
            True
        )

        text(
            "M: move/wall   H/V: wall direction   R: reset   1/2/3: AI level",
            WIDTH // 2,
            85,
            tiny_font,
            DIM,
            True
        )

        pygame.draw.line(
            screen,
            WHITE,
            (38, 105),
            (562, 105),
            1
        )

        # --------------------
        # AI INFORMATION
        # --------------------

        pygame.draw.circle(
            screen,
            RED,
            (167, 126),
            9
        )

        text(
            "AI",
            184,
            116,
            label_font
        )

        text(
            f"{ai_kind.upper()}  |  WALLS: {state.remaining[1]}  |  DATA: {min(completed_count(), DATA_TARGET)}/{DATA_TARGET}",
            229,
            119,
            info_font
        )

        # --------------------
        # BOARD
        # --------------------

        for row in range(SIZE):
            for col in range(SIZE):

                pygame.draw.rect(
                    screen,
                    WHITE,
                    tile_rect(row, col),
                    1,
                    border_radius=5
                )

        # --------------------
        # VALID MOVES
        # --------------------

        if (
            state.turn == 0
            and state.winner is None
            and not wall_mode
        ):

            for row, col in legal_pawn_moves(state):

                pygame.draw.circle(
                    screen,
                    GREEN,
                    tile_rect(row, col).center,
                    6
                )

        # --------------------
        # PLACED WALLS
        # --------------------

        for orient, row, col in sorted(state.walls):

            pygame.draw.rect(
                screen,
                GREEN,
                wall_rect(orient, row, col)
            )

        # --------------------
        # WALL PREVIEW
        # --------------------

        if (
            wall_mode
            and state.turn == 0
            and state.winner is None
        ):

            anchor = nearest_anchor(
                pygame.mouse.get_pos()
            )

            if anchor is not None:

                row, col = anchor

                preview = wall_rect(
                    orientation,
                    row,
                    col
                )

                valid = can_place_wall(
                    state,
                    orientation,
                    row,
                    col
                )

                preview_surface = pygame.Surface(
                    preview.size,
                    pygame.SRCALPHA
                )

                preview_surface.fill(
                    (*GREEN, 150)
                    if valid
                    else (*RED, 135)
                )

                screen.blit(
                    preview_surface,
                    preview.topleft
                )

        # --------------------
        # PLAYER PIECES
        # --------------------

        for player, color in (
            (0, BLUE),
            (1, RED)
        ):

            center = tile_rect(
                *state.pawns[player]
            ).center

            pygame.draw.circle(
                screen,
                color,
                center,
                18
            )

            pygame.draw.circle(
                screen,
                WHITE,
                center,
                18,
                2
            )

        # --------------------
        # PLAYER INFORMATION
        # --------------------

        pygame.draw.circle(
            screen,
            BLUE,
            (168, 485),
            9
        )

        text(
            "YOU",
            185,
            475,
            label_font
        )

        text(
            f"WALLS: {state.remaining[0]}",
            245,
            478,
            info_font
        )

        # --------------------
        # CONTROL BUTTONS
        # --------------------

        for name, rect in BUTTONS.items():

            active = (
                (name == "move" and not wall_mode)
                or
                (name == orientation and wall_mode)
            )

            pygame.draw.rect(
                screen,
                GREEN if active else WHITE,
                rect,
                2 if active else 1
            )

            label = {
                "move": "MOVE",
                "h": "WALL H",
                "v": "WALL V"
            }[name]

            text(
                label,
                rect.centerx,
                rect.centery,
                info_font,
                GREEN if active else WHITE,
                True
            )

        # --------------------
        # STATUS
        # --------------------

        pygame.draw.line(
            screen,
            WHITE,
            (38, 565),
            (562, 565),
            1
        )

        if state.winner is not None:

            if state.winner == 0:
                message = "YOU WIN! GAME SAVED - PRESS R" if ai_kind == "heuristic" and completed_count() <= DATA_TARGET else "YOU WIN! PRESS R TO RESTART"
            else:
                message = "AI WINS! GAME SAVED - PRESS R" if ai_kind == "heuristic" and completed_count() <= DATA_TARGET else "AI WINS! PRESS R TO RESTART"

        else:
            message = status

        text(
            message,
            WIDTH // 2,
            581,
            tiny_font,
            GREEN,
            True
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
