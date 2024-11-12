import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.deep_q_learning import DQNAgent
from src.environments.line_world import LineWorld
from src.environments.grid_world import GridWorld
from src.environments.farkle import Farkle

def run_dqn_example(env_class, env_name, num_episodes=10000):
    env = env_class()
    state_size = len(env.state_vector())
    action_size = env.num_actions()
    buffer_size=10000
    learning_rate=0.0005*2
    gamma=0.99
    batch_size=16

    # Initialize DQN agent
    agent = DQNAgent(env, state_size=state_size, action_size=action_size, buffer_size=buffer_size, lr=learning_rate, gamma=gamma, batch_size=batch_size)

    # Train the agent
    scores = agent.train(num_episodes=num_episodes)

    # Plot the learning curve
    '''plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title(f"DQN Learning Curve - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig(f"dqn_{env_name.lower().replace(' ', '_')}_learning_curve.png")
    plt.close()'''

    return scores

def main():
    # Run DQN on LineWorld
    # line_world_scores = run_dqn_example(LineWorld, "Line World")

    # Run DQN on GridWorld
    grid_world_scores = run_dqn_example(GridWorld, "Grid World")

    # Run DQN on Farkle
    #farkle_scores = run_dqn_example(Farkle, "Farkle")

    # Print final average scores
    print("\nFinal Average Scores:")
    # print(f"Line World: {sum(line_world_scores[-100:]) / 100:.2f}")
    print(f"Grid World: {sum(grid_world_scores[-100:]) / 100:.2f}")
    # print(f"Farkle: {sum(farkle_scores[-100:]) / 100:.2f}")

if __name__ == "__main__":
    main()
