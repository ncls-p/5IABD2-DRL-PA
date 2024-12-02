# Algorithms Used in This Project

This document provides an in-depth overview of the algorithms implemented in this project.

## Tabular Q-Learning

**Overview:**

Tabular Q-Learning is a foundational model-free reinforcement learning algorithm. It aims to learn the optimal policy by estimating the value of taking a particular action in a given state and following the optimal policy thereafter. The value function is represented using a Q-table.

**Algorithm Steps:**

1. **Initialize** the Q-table with arbitrary values for all state-action pairs.
2. **For each episode:**
   - **Reset** the environment to obtain the initial state \( s_0 \).
   - **While** the terminal state is not reached:
     - **Select an action** \( a \) using an exploration strategy (e.g., ε-greedy policy).
     - **Perform the action** and observe the next state \( s' \) and reward \( r \).
     - **Update the Q-value** for the state-action pair using the Q-learning update rule:
       \[
       Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
       \]
     - **Set** the next state \( s' \) as the current state \( s \).
3. **Repeat** until convergence.

**Key Components:**

- **Learning Rate (\( \alpha \))**: Determines how much new information overrides old information.
- **Discount Factor (\( \gamma \))**: Balances immediate and future rewards.
- **Exploration Strategy**: Methods like ε-greedy balance exploration and exploitation.

**Advantages:**

- Simple and easy to implement.
- Proven convergence properties in certain conditions.

**Limitations:**

- Inefficient in large state spaces due to the need to store and update the Q-table.
- Not suitable for continuous state or action spaces.

---

## Deep Q-Network (DQN)

**Overview:**

DQN extends Q-Learning to handle high-dimensional and continuous state spaces by using deep neural networks to approximate the Q-function instead of a Q-table.

**Algorithm Steps:**

1. **Initialize** the replay memory \( D \) and the Q-network with random weights.
2. **For each episode:**
   - **Reset** the environment to obtain the initial state \( s_0 \).
   - **While** the terminal state is not reached:
     - **Select an action** \( a \) using an ε-greedy policy based on the Q-network.
     - **Execute the action** and observe \( r, s' \).
     - **Store** the transition \( (s, a, r, s') \) in the replay memory \( D \).
     - **Sample** a random minibatch of transitions from \( D \).
     - **Compute the target** \( y \):
       \[
       y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
       \]
     - **Update the Q-network** by minimizing the loss:
       \[
       L = \left( y - Q(s, a; \theta) \right)^2
       \]
     - **Set** \( s = s' \).
   - **Periodically update** the target network weights.

**Key Components:**

- **Experience Replay**: Stores past experiences to break correlation and improve learning stability.
- **Target Network**: A separate network used to compute the target Q-values, updated less frequently to stabilize training.

**Advantages:**

- Handles large and continuous state spaces.
- Experience replay improves data efficiency.

**Limitations:**

- Training can be unstable and sensitive to hyperparameters.
- Requires significant computational resources.

---

## Double DQN

**Overview:**

Double DQN addresses the overestimation bias inherent in standard DQN by decoupling the action selection from the target Q-value evaluation.

**Algorithm Steps:**

1. **Initialize** the Q-network and the target network.
2. **For each step:**
   - **Select an action** \( a \) using the current Q-network.
   - **Execute the action** and observe \( r, s' \).
   - **Store** the transition \( (s, a, r, s') \) in the replay memory.
   - **Sample** a minibatch from the replay memory.
   - **Compute** the target using Double DQN update:
     \[
     a_{\text{max}} = \arg\max_{a'} Q(s', a'; \theta)
     \]
     \[
     y = r + \gamma Q_{\text{target}}(s', a_{\text{max}}; \theta^{-})
     \]
   - **Update** the Q-network by minimizing the loss \( L = (y - Q(s, a; \theta))^2 \).
3. **Periodically update** the target network.

**Key Components:**

- **Decoupled Evaluation**: Action selection uses the current network, but evaluation uses the target network.
- **Reduced Bias**: Mitigates the overestimation of Q-values.

**Advantages:**

- More accurate value estimates.
- Improved learning performance over standard DQN.

**Limitations:**

- Added complexity over standard DQN.
- Requires careful tuning of training parameters.

---

## DQN with Experience Replay

**Overview:**

This variant enhances DQN by using a replay buffer to store past experiences and sample them randomly during training, which breaks the correlation between sequential samples.

**Key Concepts:**

- **Replay Buffer**: A finite-sized buffer storing transitions \( (s, a, r, s') \).
- **Random Sampling**: Minibatches are sampled uniformly to reduce variance and correlation.

**Benefits:**

- **Stability**: Reduces the variance of updates.
- **Efficiency**: Reuses past experiences, improving data efficiency.

---

## DQN with Prioritized Experience Replay

**Overview:**

Prioritized Experience Replay improves upon standard experience replay by sampling transitions with higher temporal-difference (TD) errors more frequently, focusing on experiences that can provide more learning progress.

**Algorithm Details:**

- **Priority Assignment**: Assigns priorities to experiences based on the magnitude of their TD-error.
- **Sampling Probability**:
  \[
  P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
  \]
  where \( p_i \) is the priority of transition \( i \), and \( \alpha \) determines the level of prioritization.
- **Importance-Sampling Weights**:
  \[
  w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta
  \]
  where \( \beta \) corrects for the bias introduced by prioritized sampling.

**Advantages:**

- Focuses on learning from more significant experiences.
- Accelerates convergence.

**Limitations:**

- Additional computational overhead.
- Complexity in implementing the priority sampling.

---

## REINFORCE

**Overview:**

REINFORCE is a Monte Carlo policy gradient method that directly optimizes the policy to maximize expected returns by following the gradient of the performance measure with respect to the policy parameters.

**Algorithm Steps:**

1. **Initialize** policy network with parameters \( \theta \).
2. **For each episode:**
   - **Generate an episode** by following the current policy.
   - **For each time step \( t \):**
     - **Compute the return** \( G_t = \sum_{k=t}^{T} \gamma^{k - t} r_k \).
     - **Update the policy parameters**:
       \[
       \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) G_t
       \]

**Key Concepts:**

- **Policy Gradient**: Updates policy parameters in the direction that increases expected rewards.
- **Episode-based Updates**: Uses entire episodes for updates, which can lead to high variance.

**Advantages:**

- Applicable to continuous action spaces.
- Does not require a value function or model of the environment.

**Limitations:**

- High variance in gradient estimates.
- Requires a large number of samples for convergence.

---

## REINFORCE with Baseline

**Overview:**

This method introduces a baseline to reduce the variance of the gradient estimates in the REINFORCE algorithm. The baseline is typically a value function that approximates the expected return from a state.

**Algorithm Modifications:**

- **Compute Advantage**:
  \[
  A_t = G_t - b(s_t)
  \]
  where \( b(s_t) \) is the baseline value for state \( s_t \).
- **Update Policy Parameters**:
  \[
  \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) A_t
  \]

**Advantages:**

- **Variance Reduction**: By subtracting the baseline, the variance of updates is reduced.
- **Improved Convergence**: Leads to faster and more stable learning.

**Baseline Choices:**

- **Constant Baseline**: Simple but less effective.
- **State-Value Function**: Learned using methods like temporal difference (TD) learning.

---

## Proximal Policy Optimization (PPO)

**Overview:**

PPO is a policy gradient method that maintains a balance between exploration and exploitation by limiting the size of policy updates. It uses a clipped surrogate objective function to prevent drastic policy changes.

**Algorithm Steps:**

1. **Collect Trajectories**: Run the current policy to collect data.
2. **Compute Advantages**: Estimate advantages using methods like Generalized Advantage Estimation (GAE).
3. **Optimize the Policy**: Update the policy by maximizing the PPO objective:
   \[
   L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
   \]
   where \( r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)} \).

**Key Components:**

- **Clipped Objective**: Prevents large policy updates by clipping the probability ratio.
- **Advantages Estimation**: Crucial for good performance; often uses GAE.

**Advantages:**

- **Sample Efficiency**: Better use of collected data.
- **Stability**: More stable training compared to traditional policy gradient methods.

**Limitations:**

- Sensitive to hyperparameters like the clipping parameter \( \epsilon \).
- Computationally intensive due to multiple epochs of updates.

---

## Monte Carlo Tree Search (MCTS)

**Overview:**

MCTS is a heuristic search algorithm used for decision-making processes, especially in game-playing AI. It builds a search tree incrementally and uses random simulations to evaluate the potential of moves.

**Algorithm Steps:**

1. **Selection**: Starting from the root node, select child nodes according to a tree policy until a leaf node is reached.
2. **Expansion**: If the leaf node is not a terminal state, expand the tree by adding one or more child nodes.
3. **Simulation (Rollout)**: Perform a random simulation from the new node to a terminal state to estimate the value.
4. **Backpropagation**: Update the value estimates and visit counts of nodes on the path back to the root node.

**Key Components:**

- **Upper Confidence Bound for Trees (UCT)**:
  \[
  UCT = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}
  \]
  where \( w_i \) is the total value of node \( i \), \( n_i \) is the visit count, \( N \) is the total visits of the parent, and \( c \) is the exploration parameter.

**Advantages:**

- Balances exploration and exploitation effectively.
- Does not require a heuristic evaluation function.

**Limitations:**

- Computationally expensive due to the large number of simulations.
- Performance depends on the accuracy of simulations and the depth of the search.

---

**Note:** Each algorithm implementation in the project is tailored to specific environments and may include additional optimizations and customizations for improved performance.
