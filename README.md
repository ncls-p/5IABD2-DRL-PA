# Reinforcement Learning Project

This project implements various reinforcement learning algorithms and environments for educational purposes.

## Structure

- `src/`: Contains the main source code.
  - `environments/`: Implementations of different game environments.
  - `agents/`: Different RL agent implementations.
  - `metrics/`: Code for calculating and storing performance metrics.
  - `ui/`: User interface for visualizing games.
  - `utils/`: Common utility functions.
- `tests/`: Contains unit tests.
- `examples/`: Contains example scripts for training agents and playing games.

## Getting Started

1. Ensure you have Python 3.12.4 installed on your system.
2. Clone the repository
3. Create a virtual environment: `python3.12 -m venv .venv`
4. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS and Linux: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run tests: `pytest tests/`

## Usage

To run the project:

1. Activate the virtual environment (if not already activated)
2. Navigate to the project root directory
3. Run a specific example or script, e.g.:
   ```
   python examples/train_agent.py
   ```
   or
   ```
   python examples/play_game.py
   ```

(More detailed usage instructions to be added as the project develops)