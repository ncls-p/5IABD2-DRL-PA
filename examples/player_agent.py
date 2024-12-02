import os
import re
import sys
import time
from typing import Type

import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe
from src.metrics.performance_metrics import PerformanceMetrics

console = Console()


class RandomAgent:
    def choose_action(self, available_actions: np.ndarray) -> int:
        return np.random.choice(available_actions)


class PlayerAgent:
    def choose_action(self, available_actions: np.ndarray, env) -> int:
        if isinstance(env, Farkle):
            return self.choose_farkle_action(available_actions, env)
        elif isinstance(env, TicTacToe):
            return self.choose_tic_tac_toe_action(available_actions)
        elif isinstance(env, LineWorld):
            return self.choose_line_world_action(available_actions)
        elif isinstance(env, GridWorld):
            return self.choose_grid_world_action(available_actions)
        else:
            return self.choose_default_action(available_actions)

    def choose_line_world_action(self, available_actions: np.ndarray) -> int:
        console.print("[bold cyan]Enter direction (Q: left, D: right):[/]")
        while True:
            action = console.input("[bold cyan]Your move: [/]").lower()
            if action == "q" and 0 in available_actions:
                return 0  # Left
            elif action == "d" and 1 in available_actions:
                return 1  # Right
            else:
                console.print(
                    "[bold red]Invalid move. Please choose a valid direction.[/]"
                )

    def choose_zqsd_action(self, available_actions: np.ndarray, env) -> int:
        console.print(
            "[bold cyan]Enter direction (Z: up, Q: left, S: down, D: right):[/]"
        )
        while True:
            action = console.input("[bold cyan]Your move: [/]").lower()
            if action == "z" and 0 in available_actions:
                return 0  # Up
            elif action == "q" and 3 in available_actions:
                return 3  # Left
            elif action == "s" and 2 in available_actions:
                return 2  # Down
            elif action == "d" and 1 in available_actions:
                return 1  # Right
            else:
                console.print(
                    "[bold red]Invalid move. Please choose a valid direction.[/]"
                )

    def choose_grid_world_action(self, available_actions: np.ndarray) -> int:
        return self.choose_zqsd_action(available_actions, None)

    def choose_farkle_action(self, available_actions: np.ndarray, env: Farkle) -> int:
        """
        Choose an action for the Farkle game.
        """
        if not hasattr(env, "dice_visible") or not env.dice_visible:
            print("\nDo you want to:")
            print(
                "-1: Bank your points (current turn score: {})".format(env.turn_score)
            )
            print(" 0: Roll the dice")
            while True:
                try:
                    choice = input("Enter your choice (-1 or 0): ")
                    action = int(choice)
                    if action in available_actions:
                        return action
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("\nChoose dice to keep (1-6) or bank points (-1)")
            print("For example: '136' keeps the 1st, 3rd, and 6th dice")
            while True:
                try:
                    choice = input("Enter your choice: ")
                    if choice == "-1":
                        return -1
                    if not choice:
                        print("Invalid choice. Please try again.")
                        continue
                    # Convert input like "136" to action number
                    kept_dice = [int(x) for x in choice if x.isdigit()]
                    if not all(1 <= x <= 6 for x in kept_dice):
                        print("Invalid dice positions. Use numbers 1-6.")
                        continue
                    # Convert to binary action
                    action = sum(2 ** (x - 1) for x in kept_dice)
                    if action in available_actions:
                        return action
                    print("Invalid combination. Try different dice.")
                except ValueError:
                    print("Invalid input. Please enter valid dice positions.")

    def choose_tic_tac_toe_action(self, available_actions: np.ndarray) -> int:
        while True:
            action = console.input("[bold cyan]Enter your action[/]: ")
            try:
                action = int(action)
                if action in available_actions:
                    return action
                else:
                    console.print(
                        "[bold red]Invalid action. Please choose from available actions.[/]"
                    )
            except ValueError:
                console.print("[bold red]Please enter a valid number.[/]")

    def choose_default_action(self, available_actions: np.ndarray) -> int:
        while True:
            action = console.input("[bold cyan]Enter your action[/]: ")
            try:
                action = int(action)
                if action in available_actions:
                    return action
                else:
                    console.print(
                        "[bold red]Invalid action. Please choose from available actions.[/]"
                    )
            except ValueError:
                console.print("[bold red]Please enter a valid number.[/]")

    @staticmethod
    def format_dice(dice: np.ndarray) -> str:
        dice_faces = [
            # Dice faces representations
            ["┌─────────┐", "│         │", "│    ●    │", "│         │", "└─────────┘"],
            [
                "┌─────────┐",
                "│  ●      │",
                "│         │",
                "│      ●  │",
                "└─────────┘",
            ],
            ["┌─────────┐", "│  ●      │", "│    ●    │", "│      ●  │", "└─────────┘"],
            ["┌─────────┐", "│  ●   ●  │", "│         │", "│  ●   ●  │", "└─────────┘"],
            ["┌─────────┐", "│  ●   ●  │", "│    ●    │", "│  ●   ●  │", "└─────────┘"],
            ["┌─────────┐", "│  ●   ●  │", "│  ●   ●  │", "│  ●   ●  │", "└─────────┘"],
        ]

        dice_display = [""] * 5
        for die in dice:
            for i in range(5):
                dice_display[i] += dice_faces[die - 1][i] + "  "

        return "\n".join(dice_display)


ENVIRONMENT_INSTRUCTIONS = {
    "LineWorld": "Enter 'Q' for left, 'D' for right to move.",
    "GridWorld": "Enter 'Z' for up, 'Q' for left, 'S' for down, 'D' for right to move.",
    "TicTacToe": "Enter a number from 0 to 8 to place your mark.\n"
    "You will be randomly assigned as ❌ (Player 1) or ⭕ (Player 2).\n"
    "The board is numbered as follows:\n"
    "0 | 1 | 2\n"
    "3 | 4 | 5\n"
    "6 | 7 | 8",
    "Farkle": "Choose dice to keep by entering their positions (1-6).\n"
    "For example, '136' keeps the 1st, 3rd, and 6th dice.\n"
    "Enter '0' to bank your points.\n"
    "You will be randomly assigned as Player 1 or Player 2.\n"
    "Aim to score points without 'farkling' (rolling no scoring dice).",
}


def display_line_world(env: LineWorld):
    table = Table(show_header=False, box=box.SQUARE)
    row = [""] * env.size
    current_state = env.state
    target_position = env.target_position
    for i in range(env.size):
        if i == current_state:
            row[i] = "🚶"  # Player icon
        elif i == target_position:
            row[i] = "🎯"  # Target icon
        else:
            row[i] = "⬜"  # Empty space
    table.add_row(*row)
    return table


def display_grid_world(env: GridWorld):
    table = Table(show_header=False, box=box.SQUARE, padding=0)
    current_state = env.state
    for y in range(env.size):
        row = []
        for x in range(env.size):
            if (x, y) == current_state:
                row.append("🚶")  # Player icon
            elif (x, y) == env.goal:
                row.append("🏁")  # Goal icon
            else:
                row.append("  ")  # Empty space
        table.add_row(*row)
        if y < env.size - 1:
            table.add_row(*["─" * 2 for _ in range(env.size)])
    return table


def display_tic_tac_toe(env: TicTacToe):
    table = Table(show_header=False, box=box.SQUARE, padding=1)
    for i in range(3):
        row = []
        for j in range(3):
            cell = env.board[i, j]
            if cell == 1:
                row.append("❌")
            elif cell == 2:
                row.append("⭕")
            else:
                row.append(Text(str(i * 3 + j), style="dim"))
        table.add_row(*row)
    return table


def play_game(
    env_class: Type[LineWorld | GridWorld | TicTacToe | Farkle], env_name: str
):
    env = env_class()
    player_agent = PlayerAgent()
    random_agent = RandomAgent()
    metrics = PerformanceMetrics()

    console.print(Panel(f"[bold green]Playing {env_name}[/]", expand=False))

    env.reset()
    done = False
    total_reward = 0
    episode_length = 0

    if isinstance(env, (TicTacToe, Farkle)):
        human_player = np.random.choice([1, 2])
        env.current_player = 1
        console.print(f"[bold cyan]You are Player {human_player}[/]")
    else:
        human_player = 1

    if isinstance(env, Farkle):
        while not done:
            display_farkle_state(env)

            if env.current_player == human_player:
                action = player_agent.choose_farkle_action(env.available_actions(), env)
            else:
                action = random_agent.choose_action(env.available_actions())
                time.sleep(1)

            _, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

    elif isinstance(env, TicTacToe):
        while not done:
            console.print(display_tic_tac_toe(env))

            # Print state vector
            console.print(f"[bold blue]State vector: {env.state_vector()}[/]")

            if env.current_player == human_player:
                console.print(
                    Panel(
                        f"Available actions: {env.available_actions()}",
                        title="Actions",
                        expand=False,
                    )
                )
                action = player_agent.choose_action(env.available_actions(), env)
                console.print(f"[bold cyan]You chose action: {action}[/]")
            else:
                action = random_agent.choose_action(env.available_actions())
                console.print(
                    f"[bold yellow]AI Player {env.current_player} chose action: {action}[/]"
                )

            # Print action vector
            console.print(f"[bold blue]Action vector: {env.action_vector(action)}[/]")

            _, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

            if done:
                winner = env.winner
                if winner:
                    if winner == human_player:
                        console.print("[bold green]You win![/]")
                    else:
                        console.print(f"[bold red]You lose. Player {winner} wins.[/]")
                else:
                    console.print("[bold yellow]It's a tie![/]")
                break
    else:
        while not done:
            if isinstance(env, LineWorld):
                console.print(display_line_world(env))
            else:
                console.print(display_grid_world(env))

            # Print state vector
            console.print(f"[bold blue]State vector: {env.state_vector()}[/]")

            console.print(
                Panel(
                    f"Available actions: {env.available_actions()}",
                    title="Actions",
                    expand=False,
                )
            )
            action = player_agent.choose_action(env.available_actions(), env)
            action_names = ["Up", "Right", "Down", "Left"]
            console.print(f"[bold cyan]You moved: {action_names[action]}[/]")

            # Print action vector
            console.print(f"[bold blue]Action vector: {env.action_vector(action)}[/]")

            _, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

        console.print("\n[bold green]Game Over![/] Final state:")
        if isinstance(env, LineWorld):
            console.print(display_line_world(env))
        else:
            console.print(display_grid_world(env))
        console.print(f"[bold blue]Total reward: {total_reward}[/]")
        console.print(f"[bold blue]Episode length: {episode_length}[/]")
        metrics.add_episode(total_reward, episode_length)
        return metrics

    console.print("\n[bold green]Game Over![/] Final state:")
    if isinstance(env, Farkle):
        display_farkle_state(env)

    console.print(f"[bold blue]Total reward: {total_reward}[/]")
    console.print(f"[bold blue]Episode length: {episode_length}[/]")

    metrics.add_episode(total_reward, episode_length)
    return metrics


def display_farkle_state(env: Farkle):
    if hasattr(env, "dice_visible") and env.dice_visible:
        dice_display = PlayerAgent.format_dice(env.dice)
    else:
        dice_display = "[bold yellow]Dice not visible - choose to roll or bank[/]"
    turn_score = f"Current turn score: {env.turn_score}"
    player1_score = f"Player 1 score: {env.scores[0]}"
    player2_score = f"Player 2 score: {env.scores[1]}"
    current_player = f"Current player: Player {env.current_player}"

    dice_panel = Panel(
        dice_display,
        title="Current Dice",
        expand=False,
        border_style="bold cyan",
    )
    turn_score_panel = Panel(
        turn_score,
        title="Turn Score",
        expand=False,
        border_style="bold magenta",
    )
    player1_score_panel = Panel(
        player1_score,
        title="Player 1 Score",
        expand=False,
        border_style="bold green",
    )
    player2_score_panel = Panel(
        player2_score,
        title="Player 2 Score",
        expand=False,
        border_style="bold green",
    )
    player_panel = Panel(
        current_player,
        title="Current Player",
        expand=False,
        border_style="bold yellow",
    )

    group = Group(
        dice_panel,
        Columns(
            [turn_score_panel, player1_score_panel, player2_score_panel, player_panel],
            expand=False,
        ),
    )
    console.print(group)


def main():
    environments = {
        "1": (LineWorld, "LineWorld"),
        "2": (GridWorld, "GridWorld"),
        "3": (TicTacToe, "TicTacToe"),
        "4": (Farkle, "Farkle"),
    }

    table = Table(title="Choose an environment to play")
    table.add_column("Key", style="cyan")
    table.add_column("Environment", style="magenta")

    for key, (_, name) in environments.items():
        table.add_row(key, name)

    console.print(table)

    choice = console.input("[bold cyan]Enter your choice (1-4): [/]").strip()

    if choice in environments:
        env_class, env_name = environments[choice]
        console.print(
            Panel(
                ENVIRONMENT_INSTRUCTIONS.get(
                    env_name, "No specific instructions available."
                ),
                title="Instructions",
                expand=False,
            )
        )
        metrics = play_game(env_class, env_name)
    else:
        console.print("[bold red]Invalid choice. Exiting.[/]")
        return

    console.print("\n[bold green]Game Statistics:[/]")
    console.print(f"[bold blue]Score: {metrics.get_average_score()}[/]")
    console.print(f"[bold blue]Episode Length: {metrics.get_average_length()}[/]")


if __name__ == "__main__":
    main()