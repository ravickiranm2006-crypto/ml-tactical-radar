"""
Random Policy Baseline and Metric Utilities.
"""

import numpy as np

def evaluate_random_policy(env, num_episodes=100, seed=42):
    """
    Evaluates random action policy.
    """
    np.random.seed(seed)
    successes = 0
    detections = 0
    timeouts = 0
    rewards = []
    path_lengths = []
    sample_trajectory = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        traj = [env.agent_pos]

        while not done:
            action = np.random.randint(5)
            obs, reward, terminated, truncated, info = env.step(action)
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

        rewards.append(ep_reward)

    return {
        'success_rate': successes / num_episodes,
        'detection_rate': detections / num_episodes,
        'timeout_rate': timeouts / num_episodes,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_path_length': float(np.mean(path_lengths)) if path_lengths else float('nan'),
        'sample_trajectory': sample_trajectory
    }
