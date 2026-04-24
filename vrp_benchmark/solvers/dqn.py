"""DQN-based CVRP construction policy.

Architecture:
  - State: [current_x, current_y, remaining_cap_norm, (x_i, y_i, demand_i_norm, visited_i) × n]
  - Action: which customer to visit next (masked for visited + infeasible)
  - Reward: negative arc distance at each step
  - Episode: one CVRP instance; done when all customers visited

The policy is a greedy construction: at each step it picks the customer with the
highest Q-value among feasible unvisited customers, returns to depot automatically
when no feasible customer remains (capacity exceeded for all remaining).

Training uses DQN with experience replay and a target network.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vrp_benchmark.data import CVRPInstance, generate_instance, route_cost


# ── Neural network ──────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """MLP Q-network: maps state → Q-values for each customer."""

    def __init__(self, n_customers: int, hidden: int = 256) -> None:
        super().__init__()
        input_dim = 3 + 4 * n_customers  # [cur_x, cur_y, rem_cap, (x,y,d,vis)×n]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_customers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Environment ──────────────────────────────────────────────────────────────

@dataclass
class VRPEnvState:
    instance: CVRPInstance
    current_node: int = 0          # 0 = depot, 1..n = customers
    remaining_cap: int = 0
    visited: list[bool] = field(default_factory=list)
    routes: list[list[int]] = field(default_factory=list)
    current_route: list[int] = field(default_factory=list)
    total_dist: float = 0.0


class VRPEnv:
    """Single-depot CVRP environment for DQN training."""

    def __init__(self, instance: CVRPInstance) -> None:
        self._inst = instance
        self._state = self._make_initial_state()

    def _make_initial_state(self) -> VRPEnvState:
        return VRPEnvState(
            instance=self._inst,
            current_node=0,
            remaining_cap=self._inst.capacity,
            visited=[False] * self._inst.n_customers,
            routes=[],
            current_route=[],
            total_dist=0.0,
        )

    def reset(self, instance: CVRPInstance | None = None) -> np.ndarray:
        if instance is not None:
            self._inst = instance
        self._state = self._make_initial_state()
        return self._encode_state()

    def _encode_state(self) -> np.ndarray:
        s = self._state
        inst = s.instance
        cur_pos = inst.depot if s.current_node == 0 else inst.coords[s.current_node - 1]
        rem_cap_norm = s.remaining_cap / inst.capacity
        features = [cur_pos[0], cur_pos[1], rem_cap_norm]
        for i in range(inst.n_customers):
            features.append(float(inst.coords[i, 0]))
            features.append(float(inst.coords[i, 1]))
            features.append(float(inst.demands[i]) / inst.capacity)
            features.append(1.0 if s.visited[i] else 0.0)
        return np.array(features, dtype=np.float32)

    def feasible_actions(self) -> list[int]:
        """Return indices (0-based) of feasible unvisited customers."""
        s = self._state
        return [
            i for i in range(s.instance.n_customers)
            if not s.visited[i] and s.instance.demands[i] <= s.remaining_cap
        ]

    def all_visited(self) -> bool:
        return all(self._state.visited)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Visit customer (action+1). Returns (next_state, reward, done)."""
        s = self._state
        customer_node = action + 1  # convert 0-based action to 1-based node

        # If no feasible customer with current capacity, auto-return to depot first
        if s.instance.demands[action] > s.remaining_cap:
            s.total_dist += s.instance.dist(s.current_node, 0)
            s.routes.append(s.current_route)
            s.current_route = []
            s.current_node = 0
            s.remaining_cap = s.instance.capacity

        dist_step = s.instance.dist(s.current_node, customer_node)
        reward = -dist_step

        s.total_dist += dist_step
        s.current_node = customer_node
        s.remaining_cap -= s.instance.demands[action]
        s.visited[action] = True
        s.current_route.append(customer_node)

        done = self.all_visited()
        if done:
            # Close final route back to depot
            s.total_dist += s.instance.dist(s.current_node, 0)
            reward -= s.instance.dist(s.current_node, 0)
            s.routes.append(s.current_route)
            s.current_route = []
            s.current_node = 0

        next_state = self._encode_state()
        return next_state, reward, done

    def get_solution(self) -> tuple[list[list[int]], float]:
        return self._state.routes, self._state.total_dist


# ── Replay buffer ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    feasible_next: list[int]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ── DQN agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """DQN agent for CVRP construction.

    Args:
        n_customers: Fixed problem size the agent is trained on.
        lr: Learning rate.
        gamma: Discount factor.
        epsilon_start / epsilon_end / epsilon_decay: Epsilon-greedy schedule.
        target_update_freq: Steps between target network syncs.
        batch_size: Replay batch size.
        device: torch device string.
    """

    def __init__(
        self,
        n_customers: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 200,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.n_customers = n_customers
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.q_net = QNetwork(n_customers).to(self.device)
        self.target_net = QNetwork(n_customers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self._steps = 0

    def select_action(self, state: np.ndarray, feasible: list[int]) -> int:
        if not feasible:
            raise ValueError("No feasible actions")
        if random.random() < self.epsilon:
            return random.choice(feasible)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(s).squeeze(0).cpu().numpy()
        # Mask infeasible actions
        mask = np.full(self.n_customers, -np.inf)
        mask[feasible] = q[feasible]
        return int(np.argmax(mask))

    def push(self, t: Transition) -> None:
        self.buffer.push(t)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            # Mask infeasible actions in next state
            for idx, t in enumerate(batch):
                if t.done:
                    next_q[idx] = torch.full((self.n_customers,), -1e9, device=self.device)
                else:
                    mask = torch.full((self.n_customers,), -1e9, device=self.device)
                    if t.feasible_next:
                        mask[t.feasible_next] = next_q[idx][t.feasible_next]
                    next_q[idx] = mask
            target = rewards + self.gamma * next_q.max(dim=1).values * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(), "n_customers": self.n_customers}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        data = torch.load(path, map_location=device, weights_only=True)
        agent = cls(n_customers=data["n_customers"], device=device)
        agent.q_net.load_state_dict(data["q_net"])
        agent.target_net.load_state_dict(data["q_net"])
        agent.epsilon = 0.0  # greedy inference
        return agent


# ── Solver wrapper ────────────────────────────────────────────────────────────

class DQNSolver:
    """Wraps a trained DQNAgent as a CVRPSolver (greedy inference, no exploration)."""

    def __init__(self, agent: DQNAgent) -> None:
        self._agent = agent
        self._agent.epsilon = 0.0

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        if instance.n_customers != self._agent.n_customers:
            raise ValueError(
                f"DQN trained for n={self._agent.n_customers}, got n={instance.n_customers}"
            )
        env = VRPEnv(instance)
        state = env.reset(instance)
        s = env._state

        while not env.all_visited():
            feasible = env.feasible_actions()
            if not feasible:
                # Current vehicle is full — return to depot and start a new route.
                s.total_dist += instance.dist(s.current_node, 0)
                if s.current_route:
                    s.routes.append(s.current_route)
                s.current_route = []
                s.current_node = 0
                s.remaining_cap = instance.capacity
                state = env._encode_state()
                continue
            action = self._agent.select_action(state, feasible)
            state, _, _ = env.step(action)

        routes, cost = env.get_solution()
        return routes, cost


def _make_batch_instances(
    n_envs: int, n_customers: int, capacity: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of random CVRP instances as tensors.

    Returns:
        coords: (n_envs, n_customers+1, 2) — depot at index 0
        demands: (n_envs, n_customers) int
        dist: (n_envs, n_customers+1, n_customers+1) float
    """
    rng = np.random.default_rng(seed)
    depot = np.full((n_envs, 1, 2), 0.5)
    cust_coords = rng.uniform(0.0, 1.0, size=(n_envs, n_customers, 2))
    coords = np.concatenate([depot, cust_coords], axis=1)  # (B, n+1, 2)
    demands = rng.integers(1, 11, size=(n_envs, n_customers))

    # Pairwise distances (B, n+1, n+1)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(demands, dtype=torch.int32),
        torch.tensor(dist, dtype=torch.float32),
    )


def _encode_batch_state(
    cur_node: torch.Tensor,      # (B,) int — current node index (0=depot)
    rem_cap: torch.Tensor,       # (B,) int
    visited: torch.Tensor,       # (B, n) bool
    coords: torch.Tensor,        # (B, n+1, 2)
    demands: torch.Tensor,       # (B, n) int
    capacity: int,
) -> torch.Tensor:
    """Encode batch state → (B, 3 + 4*n) float32."""
    B, n = visited.shape
    device = cur_node.device

    # Current position: gather from coords using cur_node index
    idx = cur_node.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)  # (B,1,2)
    cur_pos = coords.to(device).gather(1, idx).squeeze(1)  # (B, 2)
    rem_norm = rem_cap.float() / capacity  # (B,)

    cust_coords = coords[:, 1:, :].to(device)      # (B, n, 2)
    dem_norm = demands.float().to(device) / capacity  # (B, n)
    vis = visited.float()                             # (B, n)

    # Interleave per-customer features: (x, y, demand, visited) × n
    # Must match VRPEnv._encode_state() ordering so inference and training align.
    per_cust = torch.stack([
        cust_coords[:, :, 0],   # x
        cust_coords[:, :, 1],   # y
        dem_norm,               # demand_norm
        vis,                    # visited
    ], dim=2).reshape(B, -1)    # (B, 4n)

    state = torch.cat([
        cur_pos,                      # (B, 2)
        rem_norm.unsqueeze(1),        # (B, 1)
        per_cust,                     # (B, 4n)
    ], dim=1)
    return state


def train(
    n_customers: int = 50,
    n_episodes: int = 5_000,
    capacity: int = 50,
    device: str = "cpu",
    seed: int = 0,
    n_envs: int = 256,
) -> DQNAgent:
    """Train a DQN agent using fully vectorized GPU environments.

    All n_envs environments are represented as GPU tensors and stepped
    simultaneously with no Python loops per environment. One GPU update
    per step across the full batch. ~100x faster than serial training.

    Args:
        n_customers: Fixed problem size.
        n_episodes: Total training episodes across all envs.
        capacity: Vehicle capacity.
        device: torch device ('cuda' recommended).
        seed: Base random seed.
        n_envs: Parallel environments per batch.

    Returns:
        Trained DQNAgent.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = torch.device(device)
    if hasattr(train, "_last_logged"):
        del train._last_logged  # type: ignore[attr-defined]
    agent = DQNAgent(n_customers=n_customers, device=device)
    total_rewards: list[float] = []
    ep_count = 0
    batch_seed = seed

    def _reset_batch() -> tuple:
        nonlocal batch_seed
        coords, demands, dist = _make_batch_instances(n_envs, n_customers, capacity, batch_seed)
        batch_seed += n_envs
        cur_node = torch.zeros(n_envs, dtype=torch.long, device=dev)
        rem_cap = torch.full((n_envs,), capacity, dtype=torch.int32, device=dev)
        visited = torch.zeros(n_envs, n_customers, dtype=torch.bool, device=dev)
        return coords.to(dev), demands.to(dev), dist.to(dev), cur_node, rem_cap, visited

    coords, demands, dist, cur_node, rem_cap, visited = _reset_batch()
    ep_rewards = torch.zeros(n_envs, device=dev)

    while ep_count < n_episodes:
        state = _encode_batch_state(cur_node, rem_cap, visited, coords, demands, capacity)

        # Q-values and feasibility mask
        with torch.no_grad():
            q = agent.q_net(state)  # (B, n)

        # Feasibility: not visited AND demand <= remaining capacity
        feasible_mask = (~visited) & (demands <= rem_cap.unsqueeze(1))  # (B, n)
        masked_q = q.masked_fill(~feasible_mask, -1e9)

        # Epsilon-greedy: random among feasible for exploring envs
        explore = torch.rand(n_envs, device=dev) < agent.epsilon
        rand_scores = torch.rand(n_envs, n_customers, device=dev).masked_fill(~feasible_mask, -1e9)
        greedy_action = masked_q.argmax(dim=1)
        rand_action = rand_scores.argmax(dim=1)
        actions = torch.where(explore, rand_action, greedy_action)  # (B,)

        # Step: compute rewards and update state
        # Distance: dist[b, cur_node[b], actions[b]]
        prev_node = cur_node.clone()
        step_dist = dist[torch.arange(n_envs, device=dev), cur_node, actions]  # (B,)
        rewards = -step_dist
        ep_rewards += rewards

        # Update state
        cur_node = actions + 1  # customer nodes are 1-indexed in dist matrix
        visited.scatter_(1, actions.unsqueeze(1), True)
        rem_cap = rem_cap - demands[torch.arange(n_envs, device=dev), actions]

        # Auto-return to depot when no feasible customers remain
        no_feasible = ((~visited) & (demands <= rem_cap.unsqueeze(1))).any(dim=1) == False  # noqa: E712
        # Reset vehicle: return to depot, restore capacity
        return_dist = dist[torch.arange(n_envs, device=dev), cur_node, torch.zeros(n_envs, dtype=torch.long, device=dev)]
        rewards = torch.where(no_feasible & ~visited.all(dim=1), rewards - return_dist, rewards)
        rem_cap = torch.where(no_feasible, torch.full_like(rem_cap, capacity), rem_cap)
        cur_node = torch.where(no_feasible, torch.zeros_like(cur_node), cur_node)

        # Done when all customers visited
        done = visited.all(dim=1)  # (B,)
        # Final leg: return remaining open vehicles to depot
        final_dist = dist[torch.arange(n_envs, device=dev), cur_node, torch.zeros(n_envs, dtype=torch.long, device=dev)]
        rewards = torch.where(done, rewards - final_dist, rewards)

        # Compute next state for replay
        next_state = _encode_batch_state(cur_node, rem_cap, visited, coords, demands, capacity)
        next_feasible_mask = (~visited) & (demands <= rem_cap.unsqueeze(1))

        # Push all transitions to replay buffer (CPU)
        s_np = state.cpu().numpy()
        ns_np = next_state.cpu().numpy()
        acts_np = actions.cpu().numpy()
        rews_np = rewards.cpu().numpy()
        dones_np = done.cpu().numpy()
        feas_np = next_feasible_mask.cpu().numpy()

        for i in range(n_envs):
            nf = list(np.where(feas_np[i])[0]) if not dones_np[i] else []
            agent.push(Transition(s_np[i], int(acts_np[i]), float(rews_np[i]), ns_np[i], bool(dones_np[i]), nf))

        agent.update()

        # Reset done environments
        n_done = int(done.sum().item())
        if n_done > 0:
            ep_count += n_done
            done_rewards = ep_rewards[done].cpu().tolist()
            total_rewards.extend(done_rewards)
            ep_rewards[done] = 0.0

            # Regenerate instances for done envs
            done_idx = done.nonzero(as_tuple=True)[0]
            new_coords, new_demands, new_dist = _make_batch_instances(
                len(done_idx), n_customers, capacity, batch_seed
            )
            batch_seed += len(done_idx)
            coords[done_idx] = new_coords.to(dev)
            demands[done_idx] = new_demands.to(dev)
            dist[done_idx] = new_dist.to(dev)
            cur_node[done_idx] = 0
            rem_cap[done_idx] = capacity
            visited[done_idx] = False

        last_logged = getattr(train, "_last_logged", -1)
        milestone = (ep_count // 500) * 500
        if milestone > 0 and milestone > last_logged and len(total_rewards) >= 500:
            avg = np.mean(total_rewards[-500:])
            print(f"  Episode {ep_count:5d}  avg_reward={avg:8.3f}  epsilon={agent.epsilon:.3f}")
            train._last_logged = milestone  # type: ignore[attr-defined]

    return agent
