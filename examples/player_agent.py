import os
import sys
from typing import Type

import numpy as np
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe
from src.metrics.performance_metrics import PerformanceMetrics

console = Console()


class PlayerAgent:
    def choose_action(self, available_actions: np.ndarray, env) -> int:
        if isinstance(env, Farkle):
            return self.choose_farkle_action(available_actions, env.dice)
        else:
            return self.choose_default_action(available_actions)

    def choose_farkle_action(
        self, available_actions: np.ndarray, dice: np.ndarray
    ) -> int:
        while True:
            console.print(
                Panel(
                    "0: Bank points\n1-6: Keep specific dice",
                    title="Options",
                    expand=False,
                )
            )

            action = console.input(
                "[bold cyan]Enter your action[/] (0 to bank, or dice numbers to keep, e.g., '136' to keep 1st, 3rd, and 6th dice): "
            )

            if action == "0":
                return 0  # Bank points

            try:
                kept_dice = [int(d) for d in action if d in "123456"]
                kept_dice = list(set(kept_dice))  # Remove duplicates
                if not kept_dice:
                    console.print(
                        "[bold red]Invalid input. Please enter valid dice numbers.[/]"
                    )
                    continue

                binary_action = sum(2 ** (i - 1) for i in kept_dice)
                if binary_action in available_actions:
                    return binary_action
                else:
                    console.print(
                        "[bold red]Invalid action. Please choose from available actions.[/]"
                    )
            except ValueError:
                console.print("[bold red]Please enter valid numbers.[/]")

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
            # ⚀
            ["┌─────────┐", "│         │", "│    ●    │", "│         │", "└─────────┘"],
            # ⚁
            ["┌─────────┐", "│  ●      │", "│         │", "│      ●  │", "└─────────┘"],
            # ⚂
            ["┌─────────┐", "│  ●      │", "│    ●    │", "│      ●  │", "└─────────┘"],
            # ⚃
            ["┌─────────┐", "│  ●   ●  │", "│         │", "│  ●   ●  │", "└─────────┘"],
            # ⚄
            ["┌─────────┐", "│  ●   ●  │", "│    ●    │", "│  ●   ●  │", "└─────────┘"],
            # ⚅
            ["┌─────────┐", "│  ●   ●  │", "│  ●   ●  │", "│  ●   ●  │", "└─────────┘"],
        ]

        dice_display = [""] * 5
        for die in dice:
            for i in range(5):
                dice_display[i] += dice_faces[die - 1][i] + "  "

        return "\n".join(dice_display)


ENVIRONMENT_INSTRUCTIONS = {
    "LineWorld": "Move left (0) or right (1) to reach the goal.",
    "GridWorld": "Move up (0), right (1), down (2), or left (3) to reach the goal.",
    "TicTacToe": "Enter a number from 0 to 8 to place your mark.\n"
    "You are ❌ (Player 1), the AI is ⭕ (Player 2).\n"
    "The board is numbered as follows:\n"
    "0 | 1 | 2\n"
    "3 | 4 | 5\n"
    "6 | 7 | 8",
    "Farkle": "Choose dice to keep by entering their positions (1-6).\n"
    "For example, '136' keeps the 1st, 3rd, and 6th dice.\n"
    "Enter '0' to bank your points.\n"
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
                row.append("  ")  # Empty space (two spaces for better alignment)
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
    agent = PlayerAgent()
    metrics = PerformanceMetrics()

    console.print(Panel(f"[bold green]Playing {env_name}[/]", expand=False))

    env.reset()
    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        if isinstance(env, Farkle):
            display_farkle_state(env)
            action = agent.choose_farkle_action(env.available_actions(), env.dice)
        elif isinstance(env, LineWorld):
            console.print(display_line_world(env))
            action = agent.choose_action(env.available_actions(), env)
        elif isinstance(env, GridWorld):
            console.print(display_grid_world(env))
            action = agent.choose_action(env.available_actions(), env)
        elif isinstance(env, TicTacToe):
            console.print(display_tic_tac_toe(env))
            action = agent.choose_action(env.available_actions(), env)
        else:
            rendered_state = env.render()
            if rendered_state is not None:
                console.print(
                    Panel(str(rendered_state), title="Current State", expand=False)
                )
            else:
                console.print(
                    Panel("Unable to render state", title="Current State", expand=False)
                )
            action = agent.choose_action(env.available_actions(), env)

        if not isinstance(env, Farkle):
            console.print(
                Panel(
                    f"Available actions: {env.available_actions()}",
                    title="Actions",
                    expand=False,
                )
            )

        _, reward, done, _ = env.step(action)

        total_reward += reward
        episode_length += 1

        console.print(f"[bold magenta]Reward: {reward}[/]")

    console.print("\n[bold green]Game Over![/] Final state:")
    if isinstance(env, Farkle):
        display_farkle_state(env)
    elif isinstance(env, LineWorld):
        console.print(display_line_world(env))
    elif isinstance(env, GridWorld):
        console.print(display_grid_world(env))
    elif isinstance(env, TicTacToe):
        console.print(display_tic_tac_toe(env))
    else:
        rendered_state = env.render()
        if rendered_state is not None:
            console.print(Panel(str(rendered_state), title="Final State", expand=False))
        else:
            console.print(
                Panel("Unable to render final state", title="Final State", expand=False)
            )

    console.print(f"[bold blue]Total reward: {total_reward}[/]")
    console.print(f"[bold blue]Episode length: {episode_length}[/]")

    metrics.add_episode(total_reward, episode_length)
    return metrics

def display_farkle_state(env: Farkle):
    dice_display = PlayerAgent.format_dice(env.dice)
    turn_score = f"Current turn score: {env.turn_score}"
    total_score = f"Total score: {env.total_score}"

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
    total_score_panel = Panel(
        total_score,
        title="Total Score",
        expand=False,
        border_style="bold green",
    )

    group = Group(
        dice_panel, Columns([turn_score_panel, total_score_panel], expand=False)
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

    choice = console.input("[bold cyan]Enter your choice (1-4): [/]")

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
