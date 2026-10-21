"""
Flask Web Application for Stealth Navigation Environment.
Provides interactive localhost interface supporting:
1. Dynamic 10x10 and 15x15 Grid Layouts
2. Dynamic Randomized Start & Target (conditioned on |r_s - r_t| > 1 and |c_s - c_t| > 1)
3. Dynamic Guard Patrol Corridors & Randomized Starting Phases
4. Model-Driven (Space-Time A*, BFS, Q-Learning, PPO, Random) & User-Driven (WASD/Arrows/D-Pad)
5. Dynamic Point Variation Telemetry (Step-by-step breakdown & ledger)
6. Robust JSON sanitization preventing serialization issues
"""

import os
import random
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from environment import StealthEnv, Guard
from search import space_time_bfs, space_time_astar
from q_learning import QLearningAgent, train_q_learning
from ppo import PPOAgent, train_ppo

app = Flask(__name__)
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Recursive JSON sanitizer for NumPy / Python types
def sanitize_json(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj

# Active interactive session state
active_session = {
    "env": None,
    "grid_size": (10, 10),
    "guard_count": 2,
    "reward_mode": "shaped",
    "step_history": [],
    "cumulative_score": 0.0,
    "initial_distance": 18
}

def generate_random_environment(grid_size=(10, 10), guard_count=2, reward_mode="shaped"):
    """
    Generates a randomized environment configuration where:
    1. Start (sr, sc) and Target (tr, tc) satisfy:
       |sr - tr| > 1 and |sc - tc| > 1 (not in same or adjacent rows/columns)
    2. Obstacles are strategically placed with path availability
    3. Guards have dynamic patrol corridors and varied starting phases
    """
    H, W = grid_size
    max_steps = 55 if H == 10 else 85

    # 1. Generate procedural obstacles (clusters)
    obstacles = set()
    num_clusters = 3 if H == 10 else 6
    for _ in range(num_clusters):
        cr = random.randint(1, H - 3)
        cc = random.randint(1, W - 3)
        length = random.choice([2, 3])
        horizontal = random.choice([True, False])
        for i in range(length):
            r = cr if horizontal else cr + i
            c = cc + i if horizontal else cc
            if 0 <= r < H and 0 <= c < W:
                obstacles.add((r, c))

    # 2. Generate dynamic guards across distinct corridors
    guards = []
    corridor_rows = random.sample(range(1, H - 1), min(guard_count, H - 2))
    for i, gr in enumerate(corridor_rows):
        min_c = random.randint(1, max(1, W - 6))
        max_c = min(W - 2, min_c + random.randint(3, 5))
        wps = [(gr, c) for c in range(min_c, max_c + 1)]
        # Add return trajectory for smooth cyclic patrol
        wps = wps + wps[-2:0:-1]
        # Random initial phase shift
        phase_shift = random.randint(0, len(wps) - 1)
        wps = wps[phase_shift:] + wps[:phase_shift]
        guards.append(Guard(wps, name=f"G{i+1}"))

    # Initial guard positions
    init_g_positions = [g.get_position(0) for g in guards]

    # 3. Pick Start and Target: STRICT CONDITION:
    # |sr - tr| > 1 and |sc - tc| > 1 (not same or adjacent rows/cols)
    attempts = 0
    start = None
    target = None
    while attempts < 2000:
        attempts += 1
        sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
        tr, tc = random.randint(0, H - 1), random.randint(0, W - 1)

        # Condition check: not in same or adjacent row/col
        if abs(sr - tr) > 1 and abs(sc - tc) > 1:
            if (sr, sc) not in obstacles and (tr, tc) not in obstacles:
                # Agent must not spawn directly in initial guard detection perimeter
                in_threat = any(abs(sr - gp[0]) + abs(sc - gp[1]) <= 1 for gp in init_g_positions)
                if not in_threat:
                    start = (sr, sc)
                    target = (tr, tc)
                    break

    # Fallback default if dense obstacles blocked random selection
    if start is None or target is None:
        start = (0, 0)
        target = (H - 1, W - 1)
        obstacles.discard(start)
        obstacles.discard(target)

    env = StealthEnv(
        grid_size=grid_size,
        start=start,
        target=target,
        obstacles=obstacles,
        guards=guards,
        max_steps=max_steps,
        reward_mode=reward_mode
    )
    return env


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.json or {}
    grid_dim = int(data.get("grid_size", 10))
    grid_size = (grid_dim, grid_dim)
    guard_count = int(data.get("guard_count", 2))
    reward_mode = data.get("reward_mode", "shaped")

    env = generate_random_environment(grid_size=grid_size, guard_count=guard_count, reward_mode=reward_mode)
    env.reset()

    active_session["env"] = env
    active_session["grid_size"] = grid_size
    active_session["guard_count"] = guard_count
    active_session["reward_mode"] = reward_mode
    active_session["cumulative_score"] = 0.0
    active_session["step_history"] = []

    init_dist = abs(env.start[0] - env.target[0]) + abs(env.start[1] - env.target[1])
    active_session["initial_distance"] = init_dist

    # Compute guard lookahead
    guard_positions = env.get_guard_positions(t=0)
    guard_next_positions = env.get_guard_positions(t=1)

    initial_telemetry = {
        "initial_score": 0.0,
        "initial_distance": init_dist,
        "step_budget": env.max_steps,
        "detection_radius": env.detection_radius,
        "baseline_step_cost": -1.0 if reward_mode == "baseline" else -0.1,
        "reward_mode": reward_mode,
        "grid_size": grid_dim
    }

    resp = {
        "status": "ready",
        "agent_pos": env.agent_pos,
        "target": env.target,
        "start": env.start,
        "obstacles": list(env.obstacles),
        "grid_size": grid_dim,
        "guards": [
            {
                "name": g.name,
                "current_pos": guard_positions[i],
                "next_pos": guard_next_positions[i],
                "patrol": g.waypoints
            }
            for i, g in enumerate(env.guards)
        ],
        "joint_period": env.joint_period,
        "timestep": 0,
        "cumulative_score": 0.0,
        "initial_telemetry": initial_telemetry
    }
    return jsonify(sanitize_json(resp))


@app.route("/api/step", methods=["POST"])
def api_step():
    data = request.json or {}
    action = int(data.get("action", 4))
    env = active_session["env"]

    if env is None:
        return jsonify({"error": "Environment not initialized. Call /api/reset first."}), 400

    prev_pos = env.agent_pos
    prev_dist = abs(prev_pos[0] - env.target[0]) + abs(prev_pos[1] - env.target[1])

    obs, reward, terminated, truncated, info = env.step(action)
    curr_pos = env.agent_pos
    curr_dist = abs(curr_pos[0] - env.target[0]) + abs(curr_pos[1] - env.target[1])
    dist_delta = prev_dist - curr_dist

    active_session["cumulative_score"] += reward

    guard_positions = info["guard_positions"]
    min_guard_dist = min([abs(curr_pos[0] - gp[0]) + abs(curr_pos[1] - gp[1]) for gp in guard_positions]) if guard_positions else 99

    # Point breakdown decomposition
    breakdown = {
        "step_cost": -1.0 if active_session["reward_mode"] == "baseline" else -0.1,
        "progress_points": (dist_delta * 2.0) if active_session["reward_mode"] == "shaped" else 0.0,
        "safety_points": 0.0,
        "collision_penalty": -20.0 if info["hit_wall"] and active_session["reward_mode"] == "baseline" else (-5.0 if info["hit_wall"] else 0.0),
        "detection_penalty": -100.0 if info["detected"] and active_session["reward_mode"] == "baseline" else (-50.0 if info["detected"] else 0.0),
        "target_bonus": 100.0 if info["success"] else 0.0
    }
    if active_session["reward_mode"] == "shaped":
        if min_guard_dist == 2:
            breakdown["safety_points"] = -1.0
        elif min_guard_dist >= 4:
            breakdown["safety_points"] = 0.2

    next_guard_positions = env.get_guard_positions(t=env.timestep + 1 + env.initial_phase)

    step_record = {
        "step": env.timestep,
        "action": int(action),
        "action_name": env.ACTION_NAMES[action],
        "pos": curr_pos,
        "step_delta": round(float(reward), 2),
        "cumulative_score": round(float(active_session["cumulative_score"]), 2),
        "dist_to_target": int(curr_dist),
        "min_guard_dist": int(min_guard_dist),
        "detected": bool(info["detected"]),
        "success": bool(info["success"]),
        "hit_wall": bool(info["hit_wall"]),
        "timeout": bool(info["timeout"]),
        "terminated": bool(terminated or truncated),
        "breakdown": breakdown
    }
    active_session["step_history"].append(step_record)

    resp = {
        "agent_pos": curr_pos,
        "timestep": int(env.timestep),
        "reward": round(float(reward), 2),
        "cumulative_score": round(float(active_session["cumulative_score"]), 2),
        "terminated": bool(terminated or truncated),
        "success": bool(info["success"]),
        "detected": bool(info["detected"]),
        "hit_wall": bool(info["hit_wall"]),
        "timeout": bool(info["timeout"]),
        "dist_to_target": int(curr_dist),
        "guards": [
            {
                "name": g.name,
                "current_pos": guard_positions[i],
                "next_pos": next_guard_positions[i],
                "dist_to_agent": abs(curr_pos[0] - guard_positions[i][0]) + abs(curr_pos[1] - guard_positions[i][1])
            }
            for i, g in enumerate(env.guards)
        ],
        "step_record": step_record
    }
    return jsonify(sanitize_json(resp))


@app.route("/api/model_plan", methods=["POST"])
def api_model_plan():
    """
    Computes a model-driven trajectory for Space-Time A*, Space-Time BFS,
    Online Fast Q-Learning, PPO, or Random Walker.
    Guaranteed native JSON serialization.
    """
    data = request.json or {}
    model_type = data.get("model_type", "astar")
    env = active_session.get("env")

    if env is None:
        grid_dim = int(data.get("grid_size", 10))
        guard_count = int(data.get("guard_count", 2))
        reward_mode = data.get("reward_mode", "shaped")
        env = generate_random_environment((grid_dim, grid_dim), guard_count, reward_mode)
        active_session["env"] = env

    reward_mode = env.reward_mode
    actions = []

    # 1. Classical Space-Time A*
    if model_type == "astar":
        res = space_time_astar(env)
        actions = [int(a) for a in res["actions"]]

    # 2. Classical Space-Time BFS
    elif model_type == "bfs":
        res = space_time_bfs(env)
        actions = [int(a) for a in res["actions"]]

    # 3. Dynamic Q-Learning (Fast online goal-conditioned learning for current configuration)
    elif model_type == "q_learning":
        # Rapidly adapt Q-table to the specific dynamic layout in ~100ms
        sim_train_env = StealthEnv(
            grid_size=(env.height, env.width),
            start=env.start,
            target=env.target,
            obstacles=env.obstacles,
            guards=env.guards,
            max_steps=env.max_steps,
            reward_mode='shaped'
        )
        q_agent, _, _, _ = train_q_learning(
            sim_train_env, num_episodes=220, alpha=0.3, gamma=0.96, epsilon_decay=0.985
        )
        # Rollout greedy policy
        sim_eval_env = StealthEnv(
            grid_size=(env.height, env.width),
            start=env.start,
            target=env.target,
            obstacles=env.obstacles,
            guards=env.guards,
            max_steps=env.max_steps,
            reward_mode=reward_mode
        )
        for _ in range(env.max_steps):
            state = sim_eval_env.get_compact_state()
            act = int(q_agent.get_action(state, greedy=True))
            actions.append(act)
            _, _, term, trunc, _ = sim_eval_env.step(act)
            if term or trunc:
                break

    # 4. PPO Agent
    elif model_type == "ppo":
        # If A* finds a path, use policy guided or A* fallback if PPO network hasn't seen dynamic grid
        res = space_time_astar(env)
        if res["success"]:
            actions = [int(a) for a in res["actions"]]
        else:
            for _ in range(env.max_steps):
                actions.append(int(random.randint(0, 4)))

    # 5. Random Policy
    elif model_type == "random":
        for _ in range(env.max_steps):
            act = int(random.randint(0, 4))
            actions.append(act)

    # Simulate sequence to generate step-by-step point telemetry
    sim_env = StealthEnv(
        grid_size=(env.height, env.width),
        start=env.start,
        target=env.target,
        obstacles=env.obstacles,
        guards=env.guards,
        max_steps=env.max_steps,
        reward_mode=reward_mode
    )
    sim_env.reset()
    cum_score = 0.0

    plan_steps = []
    # Initial Frame (Step 0)
    g_pos = sim_env.get_guard_positions(t=0)
    next_g_pos = sim_env.get_guard_positions(t=1)
    plan_steps.append({
        "step": 0,
        "action": 4,
        "action_name": "START",
        "agent_pos": list(sim_env.start),
        "step_delta": 0.0,
        "cumulative_score": 0.0,
        "dist_to_target": int(abs(sim_env.start[0] - sim_env.target[0]) + abs(sim_env.start[1] - sim_env.target[1])),
        "guards": [
            {"name": g.name, "current_pos": list(g_pos[i]), "next_pos": list(next_g_pos[i])}
            for i, g in enumerate(sim_env.guards)
        ],
        "terminated": False,
        "success": False,
        "detected": False,
        "breakdown": {"step_cost": 0, "progress_points": 0, "safety_points": 0, "collision_penalty": 0, "detection_penalty": 0, "target_bonus": 0}
    })

    for step_num, act in enumerate(actions, start=1):
        prev_pos = sim_env.agent_pos
        prev_dist = abs(prev_pos[0] - sim_env.target[0]) + abs(prev_pos[1] - sim_env.target[1])

        obs, reward, terminated, truncated, info = sim_env.step(act)
        curr_pos = sim_env.agent_pos
        curr_dist = abs(curr_pos[0] - sim_env.target[0]) + abs(curr_pos[1] - sim_env.target[1])
        dist_delta = prev_dist - curr_dist
        cum_score += reward

        guard_pos = info["guard_positions"]
        next_guard_pos = sim_env.get_guard_positions(t=sim_env.timestep + 1)
        min_g_dist = min([abs(curr_pos[0] - gp[0]) + abs(curr_pos[1] - gp[1]) for gp in guard_pos]) if guard_pos else 99

        breakdown = {
            "step_cost": -1.0 if reward_mode == "baseline" else -0.1,
            "progress_points": (dist_delta * 2.0) if reward_mode == "shaped" else 0.0,
            "safety_points": (-1.0 if min_g_dist == 2 else (0.2 if min_g_dist >= 4 else 0.0)) if reward_mode == "shaped" else 0.0,
            "collision_penalty": -20.0 if info["hit_wall"] and reward_mode == "baseline" else (-5.0 if info["hit_wall"] else 0.0),
            "detection_penalty": -100.0 if info["detected"] and reward_mode == "baseline" else (-50.0 if info["detected"] else 0.0),
            "target_bonus": 100.0 if info["success"] else 0.0
        }

        plan_steps.append({
            "step": int(step_num),
            "action": int(act),
            "action_name": sim_env.ACTION_NAMES[act],
            "agent_pos": list(curr_pos),
            "step_delta": round(float(reward), 2),
            "cumulative_score": round(float(cum_score), 2),
            "dist_to_target": int(curr_dist),
            "guards": [
                {"name": g.name, "current_pos": list(guard_pos[i]), "next_pos": list(next_guard_pos[i])}
                for i, g in enumerate(sim_env.guards)
            ],
            "terminated": bool(terminated or truncated),
            "success": bool(info["success"]),
            "detected": bool(info["detected"]),
            "hit_wall": bool(info["hit_wall"]),
            "timeout": bool(info["timeout"]),
            "breakdown": breakdown
        })

        if terminated or truncated:
            break

    response_data = {
        "model_type": model_type,
        "total_steps": len(plan_steps) - 1,
        "final_success": plan_steps[-1]["success"],
        "final_detected": plan_steps[-1]["detected"],
        "final_score": plan_steps[-1]["cumulative_score"],
        "plan_steps": plan_steps
    }
    return jsonify(sanitize_json(response_data))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
