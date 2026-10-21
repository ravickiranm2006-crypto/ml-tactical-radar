# Stealth Navigation: Autonomous Agent Pathfinding under Dynamic Moving Guards
## Comparative Study of Classical Space-Time Search (BFS, A*) and Reinforcement Learning (Q-Learning, PPO)

**Course:** Machine Learning (23MEE305)  
**Deliverables:** Complete Python Pipeline, Localhost Web Application, Academic Report (.docx & .md)  
**Repository:** Fully Reproducible Local Pipeline  

---

## 1. Executive Summary
This project investigates autonomous pathfinding in a high-threat, dynamic discrete environment termed **"Stealth Navigation"**. An agent must navigate a $10 \times 10$ grid from **START** $(0,0)$ to **TARGET** $(9,9)$ while avoiding detection by 1 to 3 deterministic moving guards.

To avoid unfair assumptions in classical search, we model pathfinding on a **Space-Time (Time-Expanded) graph** where states are represented as $(r, c, t)$, allowing agents to perform tactical waiting (`Stay` action) behind obstacles to let guards pass.

We benchmark Classical Search (**Space-Time BFS**, **Space-Time A\***) against Reinforcement Learning (**Tabular Q-Learning** with a compact phase representation, and Actor-Critic **PPO**), alongside a **Random Policy Baseline**.

Empirical findings show:
1. **Space-Time A\*** achieves optimal $18$-step paths while expanding **$90.5\%$ fewer nodes** than BFS ($73$ vs $769$ nodes, $0.79\text{ ms}$ latency).
2. Under baseline sparse rewards, RL suffers from the exploration starvation dilemma ($0\%$ success). **Potential-Based Reward Shaping (PBRS)** completely overcomes this, achieving **$100\%$ success** for both Q-Learning ($20$ steps) and PPO ($18.0$ optimal steps).
3. The interactive web application supports both **Model-Driven** and **User-Driven** navigation with real-time dynamic score telemetry.

---

## 2. Environment Design & Mathematical Formulation

### 2.1 State Representation
1. **Compact State (Tabular Q-Learning)**:
   $$s = (r_{\text{agent}},\, c_{\text{agent}},\, \phi)$$
   where $\phi = t \pmod P$ is the patrol phase of the guards ($P = \text{lcm}(p_1, p_2, \dots) = 6$).
   - State space size: $10 \times 10 \times 6 = 600$ states.
   - Guaranteed full observability and tabular convergence.
2. **Normalized Continuous Vector (PPO)**:
   $$\mathbf{o} = \left[\frac{r}{H-1}, \frac{c}{W-1}, \frac{r_T}{H-1}, \frac{c_T}{W-1}, \frac{r_{g1}}{H-1}, \frac{c_{g1}}{W-1}, \dots, \frac{\phi}{P-1}\right] \in \mathbb{R}^{11}$$

### 2.2 Action Space
$$\mathcal{A} = \{\text{UP}: (-1, 0),\, \text{DOWN}: (+1, 0),\, \text{LEFT}: (0, -1),\, \text{RIGHT}: (0, +1),\, \text{STAY}: (0, 0)\}$$

### 2.3 Reward Formulations
- **Baseline (Sparse)**:
  $$R_{\text{base}} = \begin{cases} +100 & \text{if target reached} \\ -100 & \text{if guard detected} \\ -20 & \text{if wall collision} \\ -1 & \text{per step} \end{cases}$$
- **Potential-Based Shaped Formulation**:
  $$R_{\text{shaped}} = 2.0 \cdot (d_{\text{prev}} - d_{\text{curr}}) - 0.1 + R_{\text{safety}} + R_{\text{terminal}}$$
  where $d$ is Manhattan distance to target, and $R_{\text{safety}} = -1.0$ if $\text{dist}_{\text{guard}} = 2$, $+0.2$ if $\text{dist}_{\text{guard}} \ge 4$.

---

## 3. Empirical Benchmark Results

| Algorithm | Success Rate | Detection Rate | Path Length | Latency (ms) | Nodes / Exploration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | $0.0\%$ | $99.0\%$ | Trapped | $0.30\text{ ms}$ | Uniform random walk |
| **Space-Time BFS** | **$100.0\%$** | **$0.0\%$** | **$18.0$ steps** | $6.06\text{ ms}$ | $769$ nodes expanded |
| **Space-Time A\*** | **$100.0\%$** | **$0.0\%$** | **$18.0$ steps** | **$0.79\text{ ms}$** | **$73$ nodes** ($10.5\times$ reduction) |
| **Q-Learning (Baseline)** | $0.0\%$ | $100.0\%$ | Failed | $1.67\text{ ms}$ | Sparse reward starvation |
| **Q-Learning (Shaped)** | **$100.0\%$** | **$0.0\%$** | **$20.0$ steps** | $1.67\text{ ms}$ | Monotonic convergence |
| **PPO (Baseline)** | $0.0\%$ | $100.0\%$ | Failed | $31.76\text{ ms}$ | Advantage collapse |
| **PPO (Shaped)** | **$100.0\%$** | **$0.0\%$** | **$18.0$ steps** | $31.76\text{ ms}$ | **Exact theoretical optimal** |

### Guard Scalability Study
- **1 Guard**: BFS nodes: $824$ | A* nodes: $88$ | Optimal cost: $18$ | Q-Learning: $100\%$
- **2 Guards**: BFS nodes: $769$ | A* nodes: $73$ | Optimal cost: $18$ | Q-Learning: $100\%$
- **3 Guards**: BFS nodes: $735$ | A* nodes: $60$ | Optimal cost: $18$ | Q-Learning: $100\%$

### Generalization to Unseen Guard Initial Phase
- Evaluated on shifted starting cycles $\phi_0 \in \{0, 1, 2, 3\}$: **$100.0\%$ Average Success**.

---

## 4. Web Application & Software Optimizations

The application is deployed on `http://localhost:5050` featuring:
1. **Model-Driven & User-Driven Control**: Instant toggle between autonomous algorithms (A*, BFS, Q-Learning, PPO, Random) and human manual control (WASD / Arrow Keys / On-screen Keypad).
2. **Initial Points & Dynamic Variation Ledger**:
   - Initial parameters: Score $0.0$, initial distance $18$, threat boundary $\le 1$, step budget $50$.
   - Live $\Delta \text{Points}$ decomposition: Distance progress ($\pm 2.0$), step penalty ($-0.1$), clearance bonus/warning, collisions, and terminal bonuses.
   - Real-time step ledger recording every action and score change.
3. **Space-Time Heuristic Pruning**: Instantaneous search execution ($<1\text{ ms}$).
4. **In-Memory Pretrained Cache**: Q-tables and PyTorch Actor-Critic weights cached in memory for zero inference lag.
5. **Threat Radar HUD & 1-Step Lookahead Ghost Trajectories**: Shows the 4-neighbor hazard zone around guards and ghost markers predicting where guards will move next.
