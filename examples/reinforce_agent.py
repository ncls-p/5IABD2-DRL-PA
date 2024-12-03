import os
import sys
import numpy as np
import torch
import time
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.reinforce import REINFORCEAgent
from src.environments.line_world import LineWorld
from src.environments.grid_world import GridWorld
from src.environments.tic_tac_toe import TicTacToe
from src.environments.farkle import Farkle


def create_tensorboard_writer(env_name: str) -> SummaryWriter:
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', 'reinforce', env_name, current_time)
    return SummaryWriter(log_dir)


def plot_metrics(data, title, env_name, window_size=100):
    plt.plot(data, color="gray", alpha=0.3, label="Raw")

    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    plt.plot(
        range(window_size - 1, len(data)),
        moving_avg,
        color="red",
        label=f"{window_size}-Episode Moving Average",
    )

    milestones = [1000, 5000, 10000, 50000, 100000]
    for milestone in milestones:
        if milestone <= len(data):
            avg_value = np.mean(data[max(0, milestone - 100) : milestone])
            plt.axvline(x=milestone, color="green", linestyle="--", alpha=0.3)
            plt.plot(milestone, avg_value, "go", alpha=0.6)
            plt.annotate(
                f"Milestone {milestone}\nAvg: {avg_value:.2f}",
                xy=(milestone, avg_value),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    plt.title(f"{title} - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()


def run_reinforce_example(
    env_class,
    env_name,
    num_episodes=100000,
    learning_rate=0.001,
    gamma=0.99,
    hidden_size=64
):
    env = env_class()
    state_size = len(env.state_vector())
    action_size = env.num_actions()

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backend.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"

    agent = REINFORCEAgent(
        env=env,
        state_size=state_size,
        action_size=action_size,
        lr=learning_rate,
        gamma=gamma,
        device=device,
        env_name=env_name
    )

    scores = []
    steps_per_episode = []
    action_times = []

    writer = create_tensorboard_writer(env_name)
    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "hidden_size": hidden_size,
            "num_episodes": num_episodes,
        },
        {"dummy": 0},
    )

    for episode in range(num_episodes):
        state = env.reset()
        state = env.state_vector()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_action_times = []

        states = []
        actions = []
        rewards = []
        log_probs = []

        while not done:
            start_time = time.time()
            action, log_prob = agent.choose_action(state)
            action_time = time.time() - start_time
            episode_action_times.append(action_time)

            next_state, reward, done, _ = env.step(action)
            next_state = env.state_vector()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            episode_reward += reward
            episode_steps += 1

        agent.train_episode(rewards, log_probs)

        scores.append(episode_reward)
        steps_per_episode.append(episode_steps)
        action_times.append(np.mean(episode_action_times))

        writer.add_scalar("Metrics/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Metrics/Steps_per_Episode", episode_steps, episode)
        writer.add_scalar(
            "Metrics/Average_Action_Time", np.mean(episode_action_times), episode
        )

        if len(scores) >= 100:
            writer.add_scalar(
                "Metrics/Average_100_Episodes_Reward", np.mean(scores[-100:]), episode
            )

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            avg_action_time = np.mean(action_times[-100:])
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Score: {avg_score:.2f}, "
                f"Avg Steps: {avg_steps:.2f}, "
                f"Avg Action Time: {avg_action_time:.5f}s"
            )

        if (episode + 1) % 10000 == 0:
            base_dir = os.path.join(os.getcwd(), "model", "reinforce", env_name)
            os.makedirs(base_dir, exist_ok=True)

            checkpoint_path = os.path.join(base_dir, f'checkpoint_{episode+1}.pt')

            checkpoint = {
                'episode': episode + 1,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'rewards': rewards,
                'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    final_path = os.path.join(os.getcwd(), 'model', 'reinforce',
                             env_name, 'final_model.pt')
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(agent.policy.state_dict(), final_path)

    writer.close()
    return scores, steps_per_episode, action_times


def print_scores_at_milestones(scores, env_name):
    milestones = [1000, 5000, 10000, 20000, 50000, 100000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = np.mean(scores[max(0, milestone - 100):milestone])
            print(f"Average score at episode {milestone}: {avg_score:.2f}")


def main():
    os.makedirs("src/metrics/plot/reinforce", exist_ok=True)

    env_configs = {
        # (LineWorld, "Line World"): {
        #     "num_episodes": 100000,
        #     "learning_rate": 0.0001,
        #     "gamma": 0.99,
        # },
        # (GridWorld, "Grid World"): {
        #     "num_episodes": 100000,
        #     "learning_rate": 0.00001,
        #     "learning_rate": 0.000001,
        #     "gamma": 0.9,
        # }
        # (TicTacToe, "Tic Tac Toe"): {
        #      "num_episodes": 100000,
        #     "learning_rate": 0.0005,
        #     "gamma": 0.95,
        # }
        (Farkle, "Farkle"): {
            "num_episodes": 10000,
            "learning_rate": 0.001,
            "gamma": 0.95,
        }
    }

    for (env_class, env_name), config in env_configs.items():
        print(f"\nRunning REINFORCE on {env_name}...")
        scores, steps, times = run_reinforce_example(
            env_class,
            env_name,
            **config
        )

        plt.figure(figsize=(15, 12))

        plt.subplot(3, 1, 1)
        plot_metrics(scores, "Scores", env_name)

        plt.subplot(3, 1, 2)
        plot_metrics(steps, "Steps per Episode", env_name)

        plt.subplot(3, 1, 3)
        plot_metrics(times, "Action Time (seconds)", env_name)

        plt.tight_layout()
        plt.savefig(
            f"src/metrics/plot/reinforce/reinforce_{env_name.lower().replace(' ', '_')}_metrics.png"
        )
        plt.close()

        print_scores_at_milestones(scores, env_name)


if __name__ == "__main__":
    main()
