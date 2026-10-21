"""
Proximal Policy Optimization (PPO) for Stealth Navigation.
Clean, self-contained PyTorch implementation of Actor-Critic with GAE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=11, action_dim=5, hidden_dim=64):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return Categorical(logits=logits), value

    def get_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        dist, value = self(obs_t)
        if deterministic:
            action = torch.argmax(dist.probs).item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action)).item()
        return action, log_prob, value.item()


class PPOAgent:
    def __init__(self, obs_dim=11, action_dim=5, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, c_val=0.5, c_ent=0.01, k_epochs=4, batch_size=64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c_val = c_val
        self.c_ent = c_ent
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0.0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = len(states)
        for _ in range(self.k_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_advantages = advantages_t[batch_idx]

                dist, values = self.policy(b_states)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values.squeeze(-1), b_returns)

                loss = policy_loss + self.c_val * value_loss - self.c_ent * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()


def train_ppo(env, total_timesteps=40000, rollout_steps=2048, seed=42):
    """
    Train PPO on StealthEnv.
    Returns: agent, rewards_history, success_history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = len(env.get_vector_obs())
    action_dim = 5
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)

    timesteps_so_far = 0
    rewards_history = []
    success_history = []

    obs, _ = env.reset()

    while timesteps_so_far < total_timesteps:
        # Collect rollout
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        ep_reward = 0.0
        
        for _ in range(rollout_steps):
            action, log_prob, val = agent.policy.get_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(val)

            ep_reward += reward
            timesteps_so_far += 1

            if done:
                rewards_history.append(ep_reward)
                success_history.append(1.0 if info.get('success', False) else 0.0)
                ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap next value for GAE
        _, _, next_val = agent.policy.get_action(obs, deterministic=False)
        advantages, returns = agent.compute_gae(rewards, values, dones, next_val)
        agent.update(states, actions, log_probs, returns, advantages)

    return agent, rewards_history, success_history


def evaluate_ppo(env, agent, num_eval_episodes=50):
    """
    Evaluate PPO agent using deterministic actions.
    """
    successes = 0
    detections = 0
    timeouts = 0
    total_rewards = []
    path_lengths = []
    sample_trajectory = []

    for ep in range(num_eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        traj = [env.agent_pos]

        while not done:
            action, _, _ = agent.policy.get_action(obs, deterministic=True)
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
