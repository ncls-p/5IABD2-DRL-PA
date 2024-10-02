# Reinforcement Learning Project

This project implements various reinforcement learning algorithms and environments for educational purposes. The environments include classic games and scenarios like Farkle, Tic-Tac-Toe, GridWorld, and LineWorld. The project is designed to help understand how reinforcement learning agents interact with different environments.

## Table of Contents

- [Environments](#environments)
  - [Farkle](#farkle)
  - [Tic-Tac-Toe](#tic-tac-toe)
  - [GridWorld](#gridworld)
  - [LineWorld](#lineworld)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Playing with `player_agent.py`](#playing-with-player_agentpy)
  - [Running Random Agents with `random_agent.py`](#running-random-agents-with-random_agentpy)
  - [Training Agents](#training-agents)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Environments

### Farkle

Farkle is a dice game where players aim to be the first to score 10,000 points. The game involves rolling six dice and deciding which scoring combinations to keep to accumulate points.

#### Farkle Rules

##### Setup
- Players: 2 or more
- Components: 6 dice, paper, and pencil for scoring

##### Gameplay
1. **Starting a Turn**: Players take turns rolling all six dice.
2. **Scoring Dice**: After each roll, the player must set aside at least one scoring die or combination.
3. **Decision**: The player can choose to:
   - **Bank Points**: End their turn and add the accumulated points to their total score.
   - **Continue Rolling**: Roll the remaining dice to potentially score more points.
4. **Hot Dice**: If a player scores with all six dice, they have "hot dice" and may roll all six dice again in the same turn.
5. **Farkle**: If no scoring dice are rolled, it's a "farkle," and the player loses all points accumulated in that turn.
6. **Winning the Game**: The first player to reach or exceed 10,000 points triggers the final round. All other players get one last turn to try to surpass the high score.

##### Scoring
- Single Dice:
  - 1: 100 points
  - 5: 50 points
- Three of a Kind:
  - Three 1s: 1,000 points
  - Three 2s: 200 points
  - Three 3s: 300 points
  - Three 4s: 400 points
  - Three 5s: 500 points
  - Three 6s: 600 points
- Four, Five, and Six of a Kind:
  - Four of a Kind: Double the score of the three of a kind (e.g., Four 2s = 400 points)
  - Five of a Kind: Quadruple the score of the three of a kind (e.g., Five 2s = 800 points)
  - Six of a Kind: Eight times the score of the three of a kind (e.g., Six 2s = 1,600 points)
- Special Combinations:
  - Straight (1-2-3-4-5-6): 1,500 points
  - Three Pairs: 1,500 points

##### Important Notes
- **Single Throw**: Each scoring combination must be achieved in a single throw.
- **Setting Aside Dice**: Players must set aside at least one scoring die or combination after each throw.
- **No Building**: Players cannot "build" on dice from previous throws (e.g., rolling two 1s, then rolling a third 1 doesn't count as three 1s).

#### Using the Farkle Environment

The Farkle environment simulates the game, allowing agents to play against each other or interact with a human player.

Key methods:
- `reset()`: Starts a new game.
- `step(action)`: Takes an action in the game (e.g., banking points or selecting dice to keep).
- `render()`: Displays the current state of the game.
- `available_actions()`: Returns the list of valid actions in the current state.

---

### Tic-Tac-Toe

Tic-Tac-Toe is a classic two-player game where players take turns marking spaces in a 3×3 grid, aiming to get three of their symbols in a row.

#### Using the Tic-Tac-Toe Environment

The Tic-Tac-Toe environment allows agents to play against each other or a human player.

Key methods:
- `reset()`: Resets the game to an empty board.
- `step(action)`: Places a mark on the board and updates the game state.
- `render()`: Displays the current board.
- `available_actions()`: Returns the list of available moves.

---

### GridWorld

GridWorld is a simple environment where an agent moves in a grid to reach a goal position.

#### Using the GridWorld Environment

Key methods:
- `reset()`: Places the agent at the starting position.
- `step(action)`: Moves the agent in the specified direction.
- `render()`: Displays the grid and the agent's position.
- `available_actions()`: Returns the list of possible movements from the current position.

---

### LineWorld

LineWorld is a linear version of GridWorld where the agent moves along a line to reach a goal.

#### Using the LineWorld Environment

Key methods:
- `reset()`: Places the agent at the starting position on the line.
- `step(action)`: Moves the agent left or right.
- `render()`: Displays the line with the agent's position.
- `available_actions()`: Returns the list of valid moves (left or right).

---

## Getting Started

### Prerequisites
- Python Version: Ensure you have Python 3.12.4 installed on your system.

### Installation Steps

1. Clone the Repository:
   ```
   git clone <repository_name>
   ```

2. Create a Virtual Environment:
   ```
   python -m venv venv
   ```

3. Activate the Virtual Environment:
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

4. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run Tests:
   ```
   python -m pytest tests/
   ```

---

## Usage

### Playing with player_agent.py

The `player_agent.py` script allows you to interactively play any of the provided environments.

To run the script: