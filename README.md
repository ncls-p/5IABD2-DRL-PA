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

#### Reward Calculation
- Banking points: `reward = turn_score / target_score`
- Keeping dice: `reward = score / target_score`
- Farkle (no scoring dice): `-1`

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

#### Reward Calculation
- Win: `1.0`
- Loss: `-1.0` (from the perspective of the other player)
- Draw: `0.0`
- Ongoing game / Invalid move: `0.0`

---

### GridWorld

GridWorld is a simple environment where an agent moves in a grid to reach a goal position.

#### Using the GridWorld Environment

Key methods:
- `reset()`: Places the agent at the starting position.
- `step(action)`: Moves the agent in the specified direction.
- `render()`: Displays the grid and the agent's position.
- `available_actions()`: Returns the list of possible movements from the current position.

#### Reward Calculation
- Reaching the goal: `1.0`
- Each step: `-1.0`
- Invalid move: `0.0`

---

### LineWorld

LineWorld is a linear version of GridWorld where the agent moves along a line to reach a goal.

#### Using the LineWorld Environment

Key methods:
- `reset()`: Places the agent at the starting position on the line.
- `step(action)`: Moves the agent left or right.
- `render()`: Displays the line with the agent's position.
- `available_actions()`: Returns the list of valid moves (left or right).

#### Reward Calculation
- Reaching the goal: `1.0`
- Each step: `-1.0`
- Invalid move: `0.0`

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
```
python player_agent.py
```

#### Instructions:
1. Choose an Environment: When prompted, enter the number corresponding to the environment you want to play.
```bash
Choose an environment to play

   ┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ Key  ┃ Environment            ┃
   ┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
   │ 1    │ LineWorld              │
   │ 2    │ GridWorld              │
   │ 3    │ TicTacToe              │
   │ 4    │ Farkle                 │
   └──────┴────────────────────────┘

   Enter your choice (1-4):
```
2. Follow Game Instructions: Each environment will display specific instructions or prompts.
3. Input Actions: Enter your moves or decisions as instructed by the game.

#### Example: Playing Farkle

- Dice Representation: The dice will be displayed along with their positions (1-6).

- Choosing Actions:
   - Bank Points: Enter 0 to bank your current turn score.
   - Keep Dice: Enter the dice numbers you wish to keep (e.g., 136 to keep dice 1, 3, and 6).
- Game Progress: The game will display updates after each action, including your score, turn score, and the current player's turn.

### Running Random Agents with random_agent.py

The random_agent.py script runs simulations where random agents play against each other or interact with the environment randomly.

To run the script:
```bash
python random_agent.py
```

What It Does:
Simulates games across different environments.
Collects performance metrics such as average score, episode length, and win rates.
Outputs statistical data for analysis.

#### Example Output
```bash
Testing Random Agents on TicTacToe
========================================
Average Score: 0.00
Average Episode Length: 5.42
Episodes per second: 12345.67
Standard Deviation: 0.99
Min Reward: -1.00
Max Reward: 1.00
Median Reward: 0.00
TicTacToe-specific statistics:
Agent1 win rate: 33.00%
Agent2 win rate: 33.00%
Tie rate: 34.00%
```

### Training Agents

You can implement and train reinforcement learning agents to interact with these environments.

#### Example:
- Implement an Agent: Create a new agent class in the src/agents/ directory.
- Train the Agent: Write a training script, such as train_agent.py, where your agent interacts with the environment.
- Run the Training Script:
```bash
python examples/train_agent.py
```

**Note:** This project provides the environments and infrastructure for training agents but does not include pre-implemented training algorithms. You are encouraged to implement algorithms like Q-learning, SARSA, DQN, etc.

---

## Project Structure

```bash
.
├── examples/
│   ├── d_dqn_agent.py
│   ├── d_dqn_exp_replay_agent.py
│   ├── d_dqn_pri_exp_replay_agent.py
│   ├── deep_q_learning_agent.py
│   ├── mcts_agent.py
│   ├── player_agent.py
│   ├── random_agent.py
│   ├── reinforce_agent.py
│   ├── reinforce_baseline_learned_by_critic.py
│   ├─�� reinforce_mean_baseline_agent.py
│   ├── tab_q_learning_agent.py
│   ├── ppo_agent.py
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── d_dqn.py
│   │   ├── d_dqn_exp_replay.py
│   │   ├── d_dqn_pri_exp_replay.py
│   │   ├── deep_q_learning.py
│   │   ├── mcts.py
│   │   ├── reinforce.py
│   │   ├── reinforce_mean_baseline.py
│   │   ├── tab_q_learning.py
│   │   ├── ppo.py
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── farkle.py
│   │   ├── grid_world.py
│   │   ├── line_world.py
│   │   ├── tic_tac_toe.py
│   ├── metrics/
│   │   └── performance_metrics.py
│   ├── ui/
│   │   └── game_viewer.py
│   ├── utils/
├── tests/
│   ├── test_farkle.py
│   ├── test_grid_world.py
│   ├── test_line_world.py
│   ├── test_mcts.py
│   ├── test_tic_tac_toe.py
├── README.md
├── requirements.txt
```
- `examples/`: Scripts demonstrating how to interact with the environments.
- `src/`:
  - `agents/`: Implementations of various RL agents (to be added by the user).
  - `environments/`: Definition of the game environments.
  - `metrics/`: Tools for tracking and analyzing performance metrics.
  - `ui/`: User interface components for game visualization.
  - `utils/`: Utility functions and helpers.
- `tests/`: Unit tests for validating the environments and other components.

## Environnements disponibles

### 1. LineWorld

Un monde linéaire où l'agent doit atteindre une cible sur une ligne.

- **Vecteur d'état** : Un tableau numpy de taille `size + 1`
  - Les `size` premiers éléments représentent la position actuelle (1 pour la position actuelle, 0 ailleurs)
  - Le dernier élément représente la position de la cible

- **Vecteur d'action** : Un tableau numpy de taille 2
  - [1, 0] pour aller à gauche
  - [0, 1] pour aller à droite

### 2. GridWorld

Un monde en grille 2D où l'agent doit atteindre une cible.

- **Vecteur d'état** : Un tableau numpy de taille `2 * size * size`
  - Les `size * size` premiers éléments représentent la position actuelle (1 pour la position actuelle, 0 ailleurs)
  - Les `size * size` éléments suivants représentent la position de la cible (1 pour la position de la cible, 0 ailleurs)

- **Vecteur d'action** : Un tableau numpy de taille 4
  - [1, 0, 0, 0] pour aller en haut
  - [0, 1, 0, 0] pour aller à droite
  - [0, 0, 1, 0] pour aller en bas
  - [0, 0, 0, 1] pour aller à gauche

### 3. TicTacToe

Le jeu classique du morpion.

- **Vecteur d'état** : Un tableau numpy de taille 9
  - Chaque élément représente une case du plateau
  - 0 pour une case vide, 1 pour le joueur 1, 2 pour le joueur 2

- **Vecteur d'action** : Un tableau numpy de taille 9
  - 1 pour la case choisie, 0 pour les autres cases

### 4. Farkle
Un jeu de dés où les joueurs doivent accumuler des points en gardant certains dés et en relançant les autres.

- **Vecteur d'état** : Un tableau numpy de taille 8

  - Les 6 premiers éléments représentent le nombre de chaque valeur de dé (1 à 6) dans le lancer actuel.
  - Le 7ᵉ élément représente le score du tour actuel.
  - Le 8ᵉ élément représente le score total du joueur actuel.

- **Vecteur d'action** : Un entier représentant l'action choisie :

  - **-1** pour banquer (conserver les points accumulés et terminer le tour).
  - **0** pour lancer les dés.
  - Un entier de **1 à 63** représentant les différentes combinaisons de dés à garder (il y a \(2^6 - 1 = 63\) combinaisons possibles en gardant au moins un dé).
