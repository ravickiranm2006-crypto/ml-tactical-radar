"""
Visualization Module for Stealth Navigation Assignment.
Generates publication-quality figures for:
1. Environment map & Guard patrol routes
2. Classical Search (BFS & A*) paths
3. RL Learning curves (Q-Learning and PPO, Baseline vs Shaped)
4. Comprehensive multi-metric benchmark comparison
5. Q-Value & Action Policy Heatmap
6. Successful vs Failed trajectory analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_environment(env, save_path=f"{OUTPUT_DIR}/env_map.png"):
    """
    Visualizes grid, obstacles, start, target, and guard patrol paths with detection buffers.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    H, W = env.height, env.width

    # Draw grid
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True, color="#d3d3d3", linestyle="--", linewidth=0.7)
    ax.invert_yaxis()  # (0,0) at top-left

    # Obstacles
    for r, c in env.obstacles:
        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#2c3e50", edgecolor="#1a252f", hatch="//")
        ax.add_patch(rect)

    # Start and Target
    ax.scatter(env.start[1], env.start[0], color="#27ae60", s=250, zorder=5, label="START (0,0)", edgecolors="black", linewidth=1.5)
    ax.scatter(env.target[1], env.target[0], color="#e74c3c", s=250, zorder=5, label=f"TARGET ({env.target[0]},{env.target[1]})", marker="*", edgecolors="black", linewidth=1.5)

    # Guards and patrol routes
    colors = ["#e67e22", "#9b59b6", "#16a085"]
    for i, guard in enumerate(env.guards):
        col = colors[i % len(colors)]
        wps = guard.waypoints
        wps_arr = np.array(wps)
        # Plot patrol route line
        ax.plot(wps_arr[:, 1], wps_arr[:, 0], color=col, linestyle="-", linewidth=2.5, alpha=0.85, label=f"{guard.name} Patrol")
        # Plot waypoints
        ax.scatter(wps_arr[:, 1], wps_arr[:, 0], color=col, s=80, edgecolors="black", zorder=4)
        # Highlight initial position
        init_pos = wps[0]
        ax.scatter(init_pos[1], init_pos[0], color=col, s=150, edgecolors="white", linewidth=2, marker="D", zorder=5)

    ax.set_title("Stealth Navigation Environment (10×10 Grid)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Column", fontsize=12)
    ax.set_ylabel("Grid Row", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_search_paths(env, bfs_res, astar_res, save_path=f"{OUTPUT_DIR}/search_paths.png"):
    """
    Plots Space-Time BFS and A* paths side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, res, title in zip(axes, [bfs_res, astar_res], ["Space-Time BFS", "Space-Time A*"]):
        H, W = env.height, env.width
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.grid(True, color="#e0e0e0", linestyle="--")
        ax.invert_yaxis()

        # Obstacles
        for r, c in env.obstacles:
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#34495e", edgecolor="#2c3e50")
            ax.add_patch(rect)

        # Start and Target
        ax.scatter(env.start[1], env.start[0], color="#27ae60", s=200, zorder=5, label="Start")
        ax.scatter(env.target[1], env.target[0], color="#e74c3c", s=220, zorder=5, marker="*", label="Target")

        # Patrol waypoints in background
        for guard in env.guards:
            wps = np.array(guard.waypoints)
            ax.plot(wps[:, 1], wps[:, 0], color="#bdc3c7", linestyle=":", linewidth=2, alpha=0.7)

        # Plot planned path
        if res['success']:
            path = np.array(res['path'])
            ax.plot(path[:, 1], path[:, 0], color="#2980b9", linestyle="-", linewidth=3.5, label="Agent Trajectory")
            for t_idx, (r, c) in enumerate(path):
                ax.text(c + 0.15, r - 0.15, str(t_idx), fontsize=8, color="#2980b9", fontweight="bold")
            subtitle = f"Cost: {res['cost']} steps | Nodes: {res['nodes_expanded']} | Time: {res['time_taken']*1000:.1f}ms"
        else:
            subtitle = "Path not found"

        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curves(q_rewards_base, q_rewards_shaped, ppo_rewards_base, ppo_rewards_shaped,
                         q_succ_base, q_succ_shaped, ppo_succ_base, ppo_succ_shaped,
                         save_path=f"{OUTPUT_DIR}/learning_curves.png"):
    """
    Plots smoothed learning curves and success rates for Q-Learning and PPO (Baseline vs Shaped).
    """
    def smooth(data, window=40):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Q-Learning Rewards
    ax = axes[0, 0]
    ax.plot(smooth(q_rewards_base), label="Baseline Reward", color="#e74c3c", linewidth=2)
    ax.plot(smooth(q_rewards_shaped), label="Shaped Reward", color="#27ae60", linewidth=2)
    ax.set_title("Q-Learning: Episode Reward", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Q-Learning Success Rate
    ax = axes[0, 1]
    ax.plot(smooth(q_succ_base), label="Baseline Reward", color="#e74c3c", linewidth=2)
    ax.plot(smooth(q_succ_shaped), label="Shaped Reward", color="#27ae60", linewidth=2)
    ax.set_title("Q-Learning: Success Rate (Moving Avg)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # PPO Rewards
    ax = axes[1, 0]
    ax.plot(smooth(ppo_rewards_base), label="Baseline Reward", color="#e67e22", linewidth=2)
    ax.plot(smooth(ppo_rewards_shaped), label="Shaped Reward", color="#3498db", linewidth=2)
    ax.set_title("PPO: Episode Reward", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # PPO Success Rate
    ax = axes[1, 1]
    ax.plot(smooth(ppo_succ_base), label="Baseline Reward", color="#e67e22", linewidth=2)
    ax.plot(smooth(ppo_succ_shaped), label="Shaped Reward", color="#3498db", linewidth=2)
    ax.set_title("PPO: Success Rate (Moving Avg)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(metrics_dict, save_path=f"{OUTPUT_DIR}/algorithm_comparison.png"):
    """
    Creates bar charts comparing Success Rate, Detection Rate, Path Length, and Runtime across all methods.
    """
    methods = list(metrics_dict.keys())
    success_rates = [metrics_dict[m].get('success_rate', 0.0) * 100 for m in methods]
    detection_rates = [metrics_dict[m].get('detection_rate', 0.0) * 100 for m in methods]
    path_lengths = [metrics_dict[m].get('path_length', 0.0) for m in methods]
    runtimes = [metrics_dict[m].get('time_ms', 0.0) for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_colors = ["#95a5a6", "#2980b9", "#16a085", "#f39c12", "#8e44ad"]

    # 1. Success Rate
    ax = axes[0, 0]
    bars = ax.bar(methods, success_rates, color=bar_colors, edgecolor="black")
    ax.set_title("Success Rate (%)", fontweight="bold")
    ax.set_ylabel("% Reached Target")
    ax.set_ylim(0, 110)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha='center', fontsize=10, fontweight="bold")
    ax.grid(axis='y', linestyle="--", alpha=0.7)

    # 2. Detection Rate
    ax = axes[0, 1]
    bars = ax.bar(methods, detection_rates, color=bar_colors, edgecolor="black")
    ax.set_title("Detection Rate (%)", fontweight="bold")
    ax.set_ylabel("% Caught by Guard")
    ax.set_ylim(0, 110)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha='center', fontsize=10, fontweight="bold")
    ax.grid(axis='y', linestyle="--", alpha=0.7)

    # 3. Path Length
    ax = axes[1, 0]
    bars = ax.bar(methods, path_lengths, color=bar_colors, edgecolor="black")
    ax.set_title("Average Path Length (Steps)", fontweight="bold")
    ax.set_ylabel("Timesteps")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha='center', fontsize=10, fontweight="bold")
    ax.grid(axis='y', linestyle="--", alpha=0.7)

    # 4. Computation Time (ms)
    ax = axes[1, 1]
    bars = ax.bar(methods, runtimes, color=bar_colors, edgecolor="black")
    ax.set_title("Inference / Search Time per Episode (ms)", fontweight="bold")
    ax.set_ylabel("Time (ms)")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f"{b.get_height():.1f}", ha='center', fontsize=10, fontweight="bold")
    ax.grid(axis='y', linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_policy_heatmap(env, q_agent, save_path=f"{OUTPUT_DIR}/policy_heatmap.png"):
    """
    Visualizes max state value V(s) = max_a Q(s, a) and preferred actions across the grid.
    """
    H, W = env.height, env.width
    value_grid = np.full((H, W), np.nan)
    action_grid = np.full((H, W), -1)

    # Representative phase t=0
    phase = 0
    for r in range(H):
        for c in range(W):
            if (r, c) in env.obstacles:
                continue
            state = (r, c, phase)
            q_vals = q_agent.q_table[state]
            if np.any(q_vals != 0):
                value_grid[r, c] = np.max(q_vals)
                action_grid[r, c] = np.argmax(q_vals)
            else:
                value_grid[r, c] = 0.0

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.viridis
    cax = ax.imshow(value_grid, cmap=cmap, origin="upper")
    fig.colorbar(cax, ax=ax, label="Max State Value V(s)")

    # Arrow symbols for actions: Up, Down, Left, Right, Stay
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "•"}

    for r in range(H):
        for c in range(W):
            if (r, c) in env.obstacles:
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#2c3e50", edgecolor="none")
                ax.add_patch(rect)
            elif (r, c) == env.target:
                ax.text(c, r, "★", ha="center", va="center", color="yellow", fontsize=18, fontweight="bold")
            else:
                act = action_grid[r, c]
                if act in arrows:
                    ax.text(c, r, arrows[act], ha="center", va="center", color="white", fontsize=14, fontweight="bold")

    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True, color="black", linestyle=":", alpha=0.3)
    ax.set_title("Learned Q-Policy Value Map & Action Directives (Phase 0)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_trajectories(env, success_traj, fail_traj, save_path=f"{OUTPUT_DIR}/trajectories.png"):
    """
    Plots side-by-side example of successful and failed/detected trajectory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, traj, title, color in zip(axes, [success_traj, fail_traj], ["Successful Trajectory (Target Reached)", "Failed Trajectory (Guard Detection / Trap)"], ["#27ae60", "#c0392b"]):
        H, W = env.height, env.width
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.grid(True, color="#e0e0e0", linestyle="--")
        ax.invert_yaxis()

        for r, c in env.obstacles:
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#34495e", edgecolor="#2c3e50")
            ax.add_patch(rect)

        ax.scatter(env.start[1], env.start[0], color="#27ae60", s=200, zorder=5, label="Start")
        ax.scatter(env.target[1], env.target[0], color="#e74c3c", s=220, zorder=5, marker="*", label="Target")

        if traj and len(traj) > 0:
            t_arr = np.array(traj)
            ax.plot(t_arr[:, 1], t_arr[:, 0], color=color, linestyle="-", linewidth=3, label="Path")
            # Mark final location
            ax.scatter(t_arr[-1, 1], t_arr[-1, 0], color="black" if color == "#c0392b" else "#27ae60", s=150, marker="X" if color == "#c0392b" else "o", zorder=6)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
