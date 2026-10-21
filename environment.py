"""
Stealth Navigation Environment
A discrete 2D grid environment where an agent must navigate from START to TARGET
while avoiding detection by deterministic moving guards.
"""

import numpy as np

class Guard:
    """
    Deterministic patrol guard moving through a fixed waypoint sequence or oscillation.
    """
    def __init__(self, waypoints, name="Guard"):
        self.waypoints = [tuple(p) for p in waypoints]
        self.period = len(self.waypoints)
        self.name = name

    def get_position(self, t):
        """Returns position at timestep t."""
        return self.waypoints[t % self.period]


class StealthEnv:
    """
    Stealth Navigation Grid Environment.
    
    Grid size: H x W (default 10x10)
    Actions: 0: Up (-1, 0), 1: Down (+1, 0), 2: Left (0, -1), 3: Right (0, +1), 4: Stay (0, 0)
    Reward Modes: 'baseline', 'shaped'
    """
    ACTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0, 0)    # Stay
    }
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    def __init__(self, 
                 grid_size=(10, 10),
                 start=(0, 0),
                 target=(9, 9),
                 obstacles=None,
                 guards=None,
                 max_steps=50,
                 reward_mode='baseline',
                 detection_radius=1):
        self.height, self.width = grid_size
        self.start = tuple(start)
        self.target = tuple(target)
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.detection_radius = detection_radius  # Manhattan distance <= 1 means detected

        # Default obstacles: strategic tactical choke points
        if obstacles is None:
            self.obstacles = {
                (2, 2), (2, 3), (2, 4),
                (4, 6), (4, 7), (4, 8),
                (6, 1), (6, 2), (6, 3),
                (7, 5), (7, 6)
            }
        else:
            self.obstacles = set(tuple(o) for o in obstacles)

        # Default guards: patrolling corridors
        if guards is None:
            self.guards = [
                Guard([(1, 1), (1, 2), (1, 3), (1, 4), (1, 3), (1, 2)], name="G1"),
                Guard([(5, 4), (5, 5), (5, 6), (5, 5)], name="G2")
            ]
        else:
            self.guards = guards

        # Calculate joint patrol period (Least Common Multiple of guard periods)
        periods = [g.period for g in self.guards] if self.guards else [1]
        self.joint_period = int(np.lcm.reduce(periods)) if len(periods) > 1 else periods[0]

        self.reset()

    def reset(self, initial_phase=0):
        """Reset environment to starting state."""
        self.agent_pos = self.start
        self.timestep = 0
        self.initial_phase = initial_phase
        return self.get_obs(), {}

    def get_guard_positions(self, t=None):
        """Get positions of all guards at time t."""
        if t is None:
            t = self.timestep + self.initial_phase
        return [g.get_position(t) for g in self.guards]

    def is_detected(self, pos, t=None):
        """Check if position is detected by any guard at time t."""
        guard_positions = self.get_guard_positions(t)
        for gp in guard_positions:
            dist = abs(pos[0] - gp[0]) + abs(pos[1] - gp[1])
            if dist <= self.detection_radius:
                return True
        return False

    def is_valid_cell(self, pos):
        """Check if cell is within grid boundaries and not an obstacle."""
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and (r, c) not in self.obstacles

    def step(self, action):
        """
        Execute one action step.
        Returns: obs, reward, terminated, truncated, info
        """
        prev_pos = self.agent_pos
        prev_target_dist = abs(prev_pos[0] - self.target[0]) + abs(prev_pos[1] - self.target[1])

        dr, dc = self.ACTIONS[action]
        next_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        self.timestep += 1

        hit_wall = False
        if not self.is_valid_cell(next_pos):
            hit_wall = True
            next_pos = self.agent_pos  # stay in place if wall/boundary hit

        self.agent_pos = next_pos
        guard_positions = self.get_guard_positions()
        detected = self.is_detected(self.agent_pos)
        reached_target = (self.agent_pos == self.target)
        timeout = (self.timestep >= self.max_steps)

        terminated = reached_target or detected
        truncated = timeout and not terminated

        # Reward computation
        curr_target_dist = abs(self.agent_pos[0] - self.target[0]) + abs(self.agent_pos[1] - self.target[1])
        reward = self._compute_reward(action, hit_wall, detected, reached_target, prev_target_dist, curr_target_dist)

        info = {
            "success": reached_target,
            "detected": detected,
            "hit_wall": hit_wall,
            "timeout": timeout,
            "timestep": self.timestep,
            "guard_positions": guard_positions
        }

        return self.get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, action, hit_wall, detected, reached_target, prev_dist, curr_dist):
        """
        Computes reward according to selected formulation.
        
        Baseline Formulation (Sparse):
          +100 target reached
          -100 guard detection
          -20 wall collision
          -1 per step
          
        Shaped Formulation (Potential-based Distance Gradient + Safety Clearance):
          Encourages progress toward target: +2 * (prev_dist - curr_dist)
          Penalizes lingering/idle steps: -0.1
          Safety clearance bonus/penalty for proximity to guards
          +100 on target arrival
          -50 on detection
          -5 on wall collision
        """
        if self.reward_mode == 'baseline':
            if reached_target:
                return 100.0
            if detected:
                return -100.0
            if hit_wall:
                return -20.0
            return -1.0  # Constant step penalty

        elif self.reward_mode == 'shaped':
            if reached_target:
                return 100.0
            if detected:
                return -50.0

            # Progress delta: positive if closer, negative if moving away
            dist_delta = float(prev_dist - curr_dist)
            reward = -0.1 + (dist_delta * 2.0)

            if hit_wall:
                reward -= 5.0

            # Proximity safety shaping
            guard_positions = self.get_guard_positions()
            if guard_positions:
                min_guard_dist = min(abs(self.agent_pos[0] - gp[0]) + abs(self.agent_pos[1] - gp[1]) for gp in guard_positions)
                if min_guard_dist == 2:
                    reward -= 1.0  # Caution zone
                elif min_guard_dist >= 4:
                    reward += 0.2  # Safe stealth corridor

            return float(reward)

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def get_obs(self):
        """Returns standard vector observation for neural network (PPO)."""
        return self.get_vector_obs()

    def get_vector_obs(self):
        """
        Normalized flat vector observation:
        [agent_r/H, agent_c/W, target_r/H, target_c/W, 
         (g1_r/H, g1_c/W), (g2_r/H, g2_c/W), ..., time_phase/period]
        """
        obs = [
            self.agent_pos[0] / (self.height - 1),
            self.agent_pos[1] / (self.width - 1),
            self.target[0] / (self.height - 1),
            self.target[1] / (self.width - 1),
        ]
        # Pad up to 3 guards for fixed-size neural network input
        guard_positions = self.get_guard_positions()
        for i in range(3):
            if i < len(guard_positions):
                gp = guard_positions[i]
                obs.extend([gp[0] / (self.height - 1), gp[1] / (self.width - 1)])
            else:
                obs.extend([-1.0, -1.0])

        phase = (self.timestep + self.initial_phase) % self.joint_period
        obs.append(phase / max(1, self.joint_period - 1))

        return np.array(obs, dtype=np.float32)

    def get_compact_state(self):
        """
        Compact hashable tuple state for Tabular Q-Learning:
        (agent_r, agent_c, patrol_phase)
        """
        phase = (self.timestep + self.initial_phase) % self.joint_period
        return (self.agent_pos[0], self.agent_pos[1], phase)
