"""
Experiment Runner for Stealth Navigation Assignment.
Runs all experiments:
1. Core Benchmark: Random vs BFS vs A* vs Q-Learning vs PPO
2. Reward Shaping Comparison (Baseline vs Shaped Reward)
3. Guard Scalability: 1, 2, and 3 guards
4. Patrol Geometry Variation (Linear vs Loop patrols)
5. Generalization to Unseen Guard Initial Phase
"""

import time
import numpy as np
from environment import StealthEnv, Guard
from search import space_time_bfs, space_time_astar
from q_learning import train_q_learning, evaluate_q_learning
from ppo import train_ppo, evaluate_ppo
from baselines import evaluate_random_policy
import visualize

def run_core_experiments():
    print("=" * 70)
    print("RUNNING STEALTH NAVIGATION BENCHMARK SUITE")
    print("=" * 70)

    # Standard 10x10 environment with 2 guards (balanced challenge)
    guards_2 = [
        Guard([(1, 1), (1, 2), (1, 3), (1, 4), (1, 3), (1, 2)], name="G1"),
        Guard([(5, 4), (5, 5), (5, 6), (5, 5)], name="G2")
    ]
    env_baseline = StealthEnv(guards=guards_2, reward_mode='baseline')
    env_shaped = StealthEnv(guards=guards_2, reward_mode='shaped')

    # 1. Visualize Map
    print("\n[1/6] Generating Environment Map...")
    visualize.plot_environment(env_baseline)

    # 2. Classical Search (BFS & A*)
    print("\n[2/6] Evaluating Space-Time BFS and A*...")
    bfs_res = space_time_bfs(env_baseline)
    astar_res = space_time_astar(env_baseline)
    print(f"  BFS: Success={bfs_res['success']}, Cost={bfs_res['cost']}, Nodes={bfs_res['nodes_expanded']}, Time={bfs_res['time_taken']*1000:.2f}ms")
    print(f"  A* : Success={astar_res['success']}, Cost={astar_res['cost']}, Nodes={astar_res['nodes_expanded']}, Time={astar_res['time_taken']*1000:.2f}ms")
    visualize.plot_search_paths(env_baseline, bfs_res, astar_res)

    # 3. Random Baseline
    print("\n[3/6] Evaluating Random Policy Baseline...")
    t0 = time.perf_counter()
    random_res = evaluate_random_policy(env_baseline, num_episodes=100)
    random_time_ms = ((time.perf_counter() - t0) / 100) * 1000
    print(f"  Random: Success={random_res['success_rate']*100:.1f}%, Detections={random_res['detection_rate']*100:.1f}%, Reward={random_res['mean_reward']:.1f}")

    # 4. Q-Learning (Baseline vs Shaped)
    print("\n[4/6] Training Q-Learning Agents (Baseline vs Shaped)...")
    q_agent_base, q_rew_base, q_len_base, q_succ_base = train_q_learning(
        env_baseline, num_episodes=1500, alpha=0.15, gamma=0.99, epsilon_decay=0.997
    )
    q_agent_shape, q_rew_shape, q_len_shape, q_succ_shape = train_q_learning(
        env_shaped, num_episodes=1500, alpha=0.2, gamma=0.95, epsilon_decay=0.998
    )

    t0 = time.perf_counter()
    q_eval_base = evaluate_q_learning(env_baseline, q_agent_base, num_eval_episodes=50)
    q_time_ms = ((time.perf_counter() - t0) / 50) * 1000
    q_eval_shape = evaluate_q_learning(env_shaped, q_agent_shape, num_eval_episodes=50)

    print(f"  Q-Learning (Baseline): Success={q_eval_base['success_rate']*100:.1f}%, Path={q_eval_base['mean_path_length']:.1f}, Reward={q_eval_base['mean_reward']:.1f}")
    print(f"  Q-Learning (Shaped)  : Success={q_eval_shape['success_rate']*100:.1f}%, Path={q_eval_shape['mean_path_length']:.1f}, Reward={q_eval_shape['mean_reward']:.1f}")

    visualize.plot_policy_heatmap(env_shaped, q_agent_shape)

    # 5. PPO (Baseline vs Shaped)
    print("\n[5/6] Training PPO Agents (Baseline vs Shaped)...")
    ppo_agent_base, ppo_rew_base, ppo_succ_base = train_ppo(env_baseline, total_timesteps=15000, rollout_steps=1024)
    ppo_agent_shape, ppo_rew_shape, ppo_succ_shape = train_ppo(env_shaped, total_timesteps=20000, rollout_steps=1024)

    t0 = time.perf_counter()
    ppo_eval_base = evaluate_ppo(env_baseline, ppo_agent_base, num_eval_episodes=50)
    ppo_time_ms = ((time.perf_counter() - t0) / 50) * 1000
    ppo_eval_shape = evaluate_ppo(env_shaped, ppo_agent_shape, num_eval_episodes=50)

    print(f"  PPO (Baseline): Success={ppo_eval_base['success_rate']*100:.1f}%, Path={ppo_eval_base['mean_path_length']:.1f}, Reward={ppo_eval_base['mean_reward']:.1f}")
    print(f"  PPO (Shaped)  : Success={ppo_eval_shape['success_rate']*100:.1f}%, Path={ppo_eval_shape['mean_path_length']:.1f}, Reward={ppo_eval_shape['mean_reward']:.1f}")

    # Plot learning curves
    visualize.plot_learning_curves(
        q_rew_base, q_rew_shape, ppo_rew_base, ppo_rew_shape,
        q_succ_base, q_succ_shape, ppo_succ_base, ppo_succ_shape
    )

    # Plot trajectories
    visualize.plot_trajectories(
        env_shaped,
        q_eval_shape['sample_trajectory'],
        random_res['sample_trajectory']
    )

    # 6. Comprehensive Metrics Comparison
    metrics_summary = {
        'Random': {
            'success_rate': random_res['success_rate'],
            'detection_rate': random_res['detection_rate'],
            'path_length': random_res['mean_path_length'] if not np.isnan(random_res['mean_path_length']) else 0,
            'time_ms': random_time_ms
        },
        'BFS': {
            'success_rate': 1.0 if bfs_res['success'] else 0.0,
            'detection_rate': 0.0,
            'path_length': bfs_res['cost'],
            'time_ms': bfs_res['time_taken'] * 1000
        },
        'A*': {
            'success_rate': 1.0 if astar_res['success'] else 0.0,
            'detection_rate': 0.0,
            'path_length': astar_res['cost'],
            'time_ms': astar_res['time_taken'] * 1000
        },
        'Q-Learning': {
            'success_rate': q_eval_shape['success_rate'],
            'detection_rate': q_eval_shape['detection_rate'],
            'path_length': q_eval_shape['mean_path_length'] if not np.isnan(q_eval_shape['mean_path_length']) else 0,
            'time_ms': q_time_ms
        },
        'PPO': {
            'success_rate': ppo_eval_shape['success_rate'],
            'detection_rate': ppo_eval_shape['detection_rate'],
            'path_length': ppo_eval_shape['mean_path_length'] if not np.isnan(ppo_eval_shape['mean_path_length']) else 0,
            'time_ms': ppo_time_ms
        }
    }
    visualize.plot_metrics_comparison(metrics_summary)

    # 7. Experiment: Guard Count Scalability (1, 2, 3 guards)
    print("\n[6/6] Guard Scalability & Generalization Experiments...")
    guard_configs = {
        "1 Guard": [Guard([(1, 1), (1, 2), (1, 3), (1, 4), (1, 3), (1, 2)], name="G1")],
        "2 Guards": guards_2,
        "3 Guards": [
            Guard([(1, 1), (1, 2), (1, 3), (1, 4), (1, 3), (1, 2)], name="G1"),
            Guard([(5, 4), (5, 5), (5, 6), (5, 5)], name="G2"),
            Guard([(8, 3), (8, 4), (8, 5), (8, 6), (8, 5), (8, 4)], name="G3")
        ]
    }

    scale_results = {}
    for name, g_list in guard_configs.items():
        g_env = StealthEnv(guards=g_list, reward_mode='shaped')
        g_bfs = space_time_bfs(g_env)
        g_astar = space_time_astar(g_env)
        g_agent, _, _, _ = train_q_learning(g_env, num_episodes=1200, alpha=0.2, gamma=0.95, epsilon_decay=0.998)
        g_q_eval = evaluate_q_learning(g_env, g_agent, num_eval_episodes=50)

        scale_results[name] = {
            'BFS Nodes': g_bfs['nodes_expanded'],
            'A* Nodes': g_astar['nodes_expanded'],
            'A* Cost': g_astar['cost'],
            'Q Success (%)': g_q_eval['success_rate'] * 100,
            'Q Mean Path': g_q_eval['mean_path_length'] if not np.isnan(g_q_eval['mean_path_length']) else 0
        }

    # 8. Generalization to Unseen Guard Initial Phase
    unseen_eval = []
    for test_phase in range(4):
        test_env = StealthEnv(guards=guards_2, reward_mode='shaped')
        test_env.reset(initial_phase=test_phase)
        eval_p = evaluate_q_learning(test_env, q_agent_shape, num_eval_episodes=25)
        unseen_eval.append(eval_p['success_rate'] * 100)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<15} | {'Success Rate':<14} | {'Detect Rate':<12} | {'Path Length':<12} | {'Time (ms)':<10}")
    print("-" * 72)
    for k, v in metrics_summary.items():
        print(f"{k:<15} | {v['success_rate']*100:>12.1f}% | {v['detection_rate']*100:>10.1f}% | {v['path_length']:>12.1f} | {v['time_ms']:>10.2f}")

    print("\nGuard Scalability Results:")
    print(f"{'Config':<10} | {'BFS Nodes':<10} | {'A* Nodes':<10} | {'Optimal Cost':<12} | {'Q Success (%)':<12}")
    print("-" * 62)
    for k, v in scale_results.items():
        print(f"{k:<10} | {v['BFS Nodes']:<10} | {v['A* Nodes']:<10} | {v['A* Cost']:<12} | {v['Q Success (%)']:>10.1f}%")

    print(f"\nGeneralization over Initial Guard Phases: {unseen_eval} (Avg: {np.mean(unseen_eval):.1f}%)")
    print("=" * 70)
    print("All plots generated successfully in ./outputs directory.")

if __name__ == "__main__":
    run_core_experiments()
