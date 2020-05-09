# openai-snake

[wip] Snake game implementation as a gym environment with a snake agent. The goal is to solve snake using a deep Q network.

# Structure
The repo is structured through a few different files.
- `env.py` is responsible for the game state. It processes new actions by the snake, updates the game board, and produces the game board when the learning system needs it.
- `snake.py` is the agent. It contains the deep Q network and is responsible for all actions within the environment.
- `train.py` is the training procedure.
- `run.py` is the task runner. It runs the training procedure and initializes the snake and environment.
- `animate.py` is an animator over a procedure similar to the task runner. It runs the training procedure currently and outputs an `anim.mp4` to visualilze the game dynamics.
-
# Bugs
There are few bugs currently within the repo, all marked with TODOs:
- The prize sometimes gets placed on top of the snake. It should be sampled in all positions except the currently occupied ones.
- The learning procedure currently discards the life ending `(state, action, reward, new_state)` pair due to inaccuracies in modeling the final states of the game. This should be fixed so that the learning procedure processes final game states, since they are in fact the highest penalizing situations to manifest.
