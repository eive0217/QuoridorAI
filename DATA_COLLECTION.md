# Human gameplay data collection (phase 1)

Run `python main.py`, choose **2** for the heuristic opponent, and finish up to **30 games**.

- Your player is **blue / Player 0**; the heuristic AI is **red / Player 1**.
- Your legal moves are recorded automatically **before** they are applied.
- At the end of each completed game, one JSON object is appended as a new line to `data/human_games.jsonl`.
- The top AI status displays `DATA: 0/30`, `1/30`, etc.
- R resets the board and **discards incomplete-game moves**.
- Matches against random (1) or neural (3) opponents are not recorded.
- Only the first 30 completed heuristic games are collected. After 30, you can still play.
- The file is saved **on your computer**, not automatically pushed to GitHub. Keep a backup if you need one.

Each record has `schema_version`, `game_id`, `opponent`, `result`, and `moves`.
Each move includes `turn`, `state` (84 numbers), `action` (0–56), and `legal_actions`.

The 84 features are player-relative: 25 own pawn cells, 25 opponent pawn cells, 16 horizontal wall anchors, 16 vertical wall anchors, and two remaining-wall ratios. Player 1's position and action are vertically reflected so the current player always aims up.

**Important:** This phase only collects training examples; it does not train the model. The current `neural_net.py` still defines the older 103→128→128→57 neural network. Model conversion and supervised training will be a separate step.

To verify records locally:
```bash
python -c "import json; from pathlib import Path; p=Path('data/human_games.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines()]; print('Games:',len(rows)); print('Moves:',sum(len(g['moves']) for g in rows)); print('State size:',len(rows[0]['moves'][0]['state']) if rows else 'none')"
```
