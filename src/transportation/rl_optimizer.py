"""
Re-Routing Optimiser

Two complementary approaches for dynamically re-allocating logistics
flows when disruptions occur:

1. **LPReRouter** — deterministic linear-programming solver (``scipy.optimize.linprog``)
   that minimises total weighted cost subject to capacity and demand constraints.
2. **ReRoutingOptimizer** — a tabular Q-learning agent (Gymnasium-compatible)
   that learns re-routing policies through simulated episodes, discovering
   allocation strategies that balance cost, transit time, and reliability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx>=3.1 required")

logger = logging.getLogger("SCRAM.Transportation.RLOptimizer")


# ====================================================================== #
#  1. Deterministic LP-based re-router
# ====================================================================== #

class LPReRouter:
    """
    Solve a minimum-cost flow problem using linear programming.

    Given a set of supply regions with (capacity, unit_cost) and a total
    demand constraint, find the allocation that minimises total cost.
    """

    def __init__(self, lane_summary: pd.DataFrame):
        """
        Parameters
        ----------
        lane_summary : pd.DataFrame
            Output of ``LogisticsNetwork.get_lane_summary()`` with at least:
            ``region, capacity, unit_cost, transit_days, reliability``.
        """
        self.lanes = lane_summary.copy()
        self.n_lanes = len(self.lanes)

    def optimise(
        self,
        demand: float,
        disrupted_regions: Optional[Dict[str, float]] = None,
        cost_weight: float = 0.6,
        time_weight: float = 0.2,
        reliability_weight: float = 0.2,
    ) -> pd.DataFrame:
        """
        Find the cost-minimising allocation.

        Parameters
        ----------
        demand : float
            Total demand (value_usd) to be satisfied.
        disrupted_regions : dict, optional
            ``{region: remaining_capacity_fraction}`` for disrupted lanes.
        cost_weight, time_weight, reliability_weight : float
            Weights for the composite objective.

        Returns
        -------
        pd.DataFrame
            Allocation per lane.
        """
        from scipy.optimize import linprog

        lanes = self.lanes.copy()

        # Apply disruptions: reduce capacity for affected regions
        if disrupted_regions:
            for region, frac in disrupted_regions.items():
                mask = lanes["region"] == region
                lanes.loc[mask, "capacity"] *= frac

        capacities = lanes["capacity"].values.astype(float)

        # Composite objective: minimise weighted sum of cost, time, inverse-reliability
        costs = lanes["unit_cost"].values.astype(float)
        times = lanes["transit_days"].values.astype(float)
        reliabilities = lanes["reliability"].values.astype(float)

        # Normalise each component to [0, 1]
        def _norm(a: np.ndarray) -> np.ndarray:
            r = a.max() - a.min()
            return (a - a.min()) / r if r > 0 else np.zeros_like(a)

        obj = (
            cost_weight * _norm(costs)
            + time_weight * _norm(times)
            + reliability_weight * (1 - _norm(reliabilities))  # lower reliability = higher penalty
        )

        # Upper bounds = per-lane capacity; sum must meet demand
        bounds = [(0, cap) for cap in capacities]
        # Equality constraint: total flow ≥ demand (use inequality form)
        # linprog minimises c^T x subject to A_ub x ≤ b_ub, A_eq x = b_eq
        # We want sum(x) >= demand  →  -sum(x) <= -demand
        A_ub = -np.ones((1, self.n_lanes))
        b_ub = np.array([-demand])

        result = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            lanes["allocation"] = result.x
        else:
            # Fallback: proportional allocation
            logger.warning("LP solver did not converge; using proportional fallback")
            total_cap = capacities.sum()
            lanes["allocation"] = (
                capacities / total_cap * demand if total_cap > 0 else 0
            )

        lanes["allocation_pct"] = (
            lanes["allocation"] / lanes["allocation"].sum() * 100
        )
        return lanes[
            ["region", "capacity", "unit_cost", "transit_days",
             "reliability", "allocation", "allocation_pct"]
        ]


# ====================================================================== #
#  2. Tabular Q-learning re-router
# ====================================================================== #

class _ReRoutingEnv:
    """
    Lightweight Gymnasium-style environment for the re-routing problem.

    State  : discretised tuple (disruption_level_per_region)
    Action : index into a set of pre-defined allocation templates
    Reward : negative of (total_cost + penalty_for_unmet_demand)
    """

    def __init__(
        self,
        lane_summary: pd.DataFrame,
        demand: float,
        n_disruption_levels: int = 4,
    ):
        self.lanes = lane_summary.reset_index(drop=True)
        self.n_lanes = len(self.lanes)
        self.demand = demand
        self.n_levels = n_disruption_levels

        # Build allocation templates (action space)
        # Each template defines what fraction of demand goes to each lane.
        self._templates = self._build_templates()
        self.n_actions = len(self._templates)

        # State space: tuple of disruption levels per lane  (0=normal, n-1=offline)
        self.state: Tuple[int, ...] = tuple([0] * self.n_lanes)

    def _build_templates(self) -> List[np.ndarray]:
        """Generate ~20 allocation templates via uniform + random combos."""
        templates = []
        n = self.n_lanes
        if n == 0:
            return [np.array([])]
        # Proportional to capacity
        caps = self.lanes["capacity"].values.astype(float)
        total = caps.sum()
        if total > 0:
            templates.append(caps / total)

        # Equal distribution
        templates.append(np.ones(n) / n)

        # Skewed: concentrate on each region in turn
        for i in range(n):
            t = np.full(n, 0.05 / max(n - 1, 1))
            t[i] = 0.95
            t = t / t.sum()
            templates.append(t)

        # Random templates
        rng = np.random.default_rng(42)
        for _ in range(10):
            t = rng.dirichlet(np.ones(n))
            templates.append(t)

        return templates

    def reset(self, disruption_state: Optional[Tuple[int, ...]] = None):
        """Reset to a disruption state (or all-normal)."""
        if disruption_state is not None:
            self.state = disruption_state
        else:
            self.state = tuple([0] * self.n_lanes)
        return self.state

    def step(self, action: int) -> Tuple:
        """Execute action, return (next_state, reward, done, info)."""
        template = self._templates[action]
        allocation = template * self.demand

        # Effective capacity after disruptions
        caps = self.lanes["capacity"].values.astype(float)
        effective_caps = np.array([
            cap * (1 - level / self.n_levels)
            for cap, level in zip(caps, self.state)
        ])

        # Clip allocation to effective capacity
        actual = np.minimum(allocation, effective_caps)
        delivered = actual.sum()
        shortfall = max(0, self.demand - delivered)

        # Cost
        costs = self.lanes["unit_cost"].values.astype(float)
        total_cost = (actual * costs).sum()

        # Reward = -cost - heavy penalty for shortfall
        reward = -(total_cost + 10 * shortfall)

        return self.state, float(reward), True, {"delivered": delivered, "shortfall": shortfall}


class ReRoutingOptimizer:
    """
    Tabular Q-learning agent for logistics re-routing.

    Trains over simulated episodes where random disruption states are
    sampled, and learns which allocation template minimises cost + shortfall
    for each disruption pattern.
    """

    def __init__(self, lane_summary: pd.DataFrame, demand: float):
        self.env = _ReRoutingEnv(lane_summary, demand)
        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.trained = False

    def _get_q(self, state: Tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.n_actions)
        return self.q_table[state]

    def train(self, n_episodes: int = 5000, seed: int = 42) -> Dict:
        """
        Train via Q-learning over random disruption states.

        Returns training summary.
        """
        rng = np.random.default_rng(seed)
        total_rewards = []

        for ep in range(n_episodes):
            # Random disruption state
            state = tuple(
                rng.integers(0, self.env.n_levels, size=self.env.n_lanes).tolist()
            )
            self.env.reset(state)

            # Epsilon-greedy action selection
            q = self._get_q(state)
            if rng.random() < self.epsilon:
                action = rng.integers(0, self.env.n_actions)
            else:
                action = int(np.argmax(q))

            _, reward, _, _ = self.env.step(action)

            # Q-update (single-step episode)
            q[action] += self.lr * (reward - q[action])
            total_rewards.append(reward)

        self.trained = True
        mean_reward = float(np.mean(total_rewards[-500:]))
        logger.info(
            "Q-learning trained for %d episodes.  Mean reward (last 500): %.2f",
            n_episodes, mean_reward,
        )
        return {
            "episodes": n_episodes,
            "q_table_size": len(self.q_table),
            "mean_reward_last_500": mean_reward,
        }

    def recommend(self, disruption_state: Tuple[int, ...]) -> pd.DataFrame:
        """
        Given a disruption state, recommend an allocation.

        Returns a DataFrame of lanes with ``allocation`` and ``allocation_pct``.
        """
        q = self._get_q(disruption_state)
        best_action = int(np.argmax(q))
        template = self.env._templates[best_action]
        allocation = template * self.env.demand

        lanes = self.env.lanes.copy()
        lanes["allocation"] = allocation
        lanes["allocation_pct"] = template * 100
        return lanes[
            ["region", "capacity", "unit_cost", "transit_days",
             "reliability", "allocation", "allocation_pct"]
        ]

    def compare_with_lp(
        self,
        disruption_state: Tuple[int, ...],
        disrupted_regions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run both RL and LP optimisers for the same disruption and return
        both allocations for comparison.
        """
        rl_result = self.recommend(disruption_state)

        lp = LPReRouter(self.env.lanes)
        lp_result = lp.optimise(
            demand=self.env.demand,
            disrupted_regions=disrupted_regions,
        )

        return {"rl": rl_result, "lp": lp_result}
