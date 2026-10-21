"""
Tabular Q-Learning Agent for Stealth Navigation.

State Representation:
Compact Markovian state tuple: (agent_r, agent_c, patrol_phase)
where patrol_phase = timestep % joint_patrol_period.
Since guard positions are deterministic functions of patrol_phase,
this 3-tuple captures full relevant observability with minimal state space.
"""

import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, num_actions=5, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.996):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: mapping state_tuple -> np.zeros(num_actions)
        self.q_table = defaultdict(lambda: np.zeros(num_actions, dtype=np.float32))

    def get_action(self, state, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_vals = self.q_table[state]
        # Break ties randomly
        max_val = np.max(q_vals)
        actions = np.where(q_vals == max_val)[0]
        return np.random.choice(actions)

    def update(self, state, action, reward, next_state, done):
        """Bellman equation update."""
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(env, num_episodes=1500, alpha=0.15, gamma=0.99, epsilon_decay=0.997, seed=42):
    """
    Train Q-Learning agent on StealthEnv.
    Returns: trained agent, episode_rewards, episode_lengths, success_history
    """
    np.random.seed(seed)
    agent = QLearningAgent(
        num_actions=5,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay
    )

    rewards_history = []
    lengths_history = []
    success_history = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = env.get_compact_state()
        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.get_action(state, greedy=False)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.get_compact_state()
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(ep_reward)
        lengths_history.append(steps)
        success_history.append(1.0 if info.get('success', False) else 0.0)

    return agent, rewards_history, lengths_history, success_history


def evaluate_q_learning(env, agent, num_eval_episodes=50):
    """
    Evaluate trained Q-Learning agent using greedy policy.
    Returns dict of evaluation metrics.
    """
    successes = 0
    detections = 0
    timeouts = 0
    total_rewards = []
    path_lengths = []
    sample_trajectory = []

    for ep in range(num_eval_episodes):
        obs, _ = env.reset()
        state = env.get_compact_state()
        ep_reward = 0.0
        done = False
        traj = [env.agent_pos]

        while not done:
            action = agent.get_action(state, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            state = env.get_compact_state()
            done = terminated or truncated
            ep_reward += reward
            traj.append(env.agent_pos)

        if ep == 0:
            sample_trajectory = traj

        if info.get('success', False):
            successes += 1
            path_lengths.append(len(traj) - 1)
        elif info.get('detected', False):
            detections += 1
        elif info.get('timeout', False):
            timeouts += 1

        total_rewards.append(ep_reward)

    return {
        'success_rate': successes / num_eval_episodes,
        'detection_rate': detections / num_eval_episodes,
        'timeout_rate': timeouts / num_eval_episodes,
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_path_length': float(np.mean(path_lengths)) if path_lengths else float('nan'),
        'sample_trajectory': sample_trajectory
    }
