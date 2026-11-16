"""Run with: python -m unittest tests.py"""
import unittest
from game import (GameState, ACTION_COUNT, apply_action, can_place_wall,
                  legal_actions, legal_pawn_moves, shortest_distance, wall_action)


class GameTests(unittest.TestCase):
    def test_start(self):
        s = GameState()
        self.assertEqual(ACTION_COUNT, 57)
        self.assertEqual(shortest_distance(s, 0), 4)
        self.assertEqual(shortest_distance(s, 1), 4)
        self.assertEqual(len(legal_pawn_moves(s)), 3)

    def test_wall_block_and_turns(self):
        s = GameState()
        a = wall_action('h', 3, 1)
        self.assertTrue(can_place_wall(s, 'h', 3, 1))
        s2 = apply_action(s, a)
        self.assertEqual(s2.remaining, [2, 3])
        self.assertIn(('h', 3, 1), s2.walls)
        self.assertEqual(s.remaining, [3, 3])
        self.assertFalse(can_place_wall(s2, 'v', 3, 1))

    def test_illegal_action(self):
        with self.assertRaises(ValueError):
            apply_action(GameState(), 0)

    def test_jump(self):
        s = GameState(pawns=[(2, 2), (1, 2)], turn=0)
        self.assertIn((0, 2), legal_pawn_moves(s))

    def test_diagonal_when_straight_jump_blocked(self):
        s = GameState(pawns=[(2, 2), (1, 2)], walls={('h', 0, 1)}, turn=0)
        legal = legal_pawn_moves(s)
        self.assertIn((1, 1), legal)
        self.assertIn((1, 3), legal)
        self.assertNotIn((0, 2), legal)

    def test_no_wall_when_empty(self):
        s = GameState(remaining=[0, 3])
        self.assertFalse(any(x >= 25 for x in legal_actions(s)))

    def test_win(self):
        s = GameState(pawns=[(1, 0), (4, 4)], turn=0)
        result = apply_action(s, 0)
        self.assertEqual(result.winner, 0)
        self.assertEqual(legal_actions(result), [])


if __name__ == '__main__':
    unittest.main()
