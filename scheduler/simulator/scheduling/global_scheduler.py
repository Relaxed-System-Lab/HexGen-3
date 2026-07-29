"""Global resource allocation search using guided simulated annealing."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from typing import Dict, List, Mapping, Optional, Tuple

from .local_scheduler import LocalScheduler
from .types import AllocationMatrix, DeploymentPlan, WORKER_TYPES, WorkloadProfile


@dataclass
class GlobalSchedulerConfig:
    iterations: int = 60
    stability_iterations: int = 12
    initial_temperature: float = 0.25
    cooling_rate: float = 0.92
    block_size: int = 0
    model_size_billions: float = 8.0
    allow_empty_source: bool = True
    seed: int = 7
    keep_history: bool = True


class GlobalScheduler:
    """Searches A by moving GPU blocks from fast workers to bottleneck workers."""

    def __init__(self, local_scheduler: LocalScheduler, config: Optional[GlobalSchedulerConfig] = None):
        self.local_scheduler = local_scheduler
        self.config = config or GlobalSchedulerConfig()
        self._rng = Random(self.config.seed)
        self._cache: Dict[Tuple, DeploymentPlan] = {}

    def optimize(
        self,
        workload: WorkloadProfile,
        capacity: Mapping[str, int],
        initial_allocation: Optional[AllocationMatrix] = None,
        previous_plan: Optional[DeploymentPlan] = None,
    ) -> DeploymentPlan:
        allocation = initial_allocation.clone() if initial_allocation else AllocationMatrix.uniform(capacity)
        allocation.validate(capacity)
        previous_parallelism = previous_plan.parallelism if previous_plan is not None else None
        current = self._evaluate(workload, allocation, previous_parallelism)
        best = current
        temperature = self.config.initial_temperature
        stable_count = 0
        history: List[Dict[str, object]] = []
        iterations_run = 0

        for iteration in range(self.config.iterations):
            iterations_run = iteration + 1
            candidate_allocation = self._generate_candidate(current.allocation, current, capacity)
            candidate_allocation.validate(capacity)
            candidate = self._evaluate(workload, candidate_allocation, current.parallelism)

            reward = candidate.throughput.bottleneck - current.throughput.bottleneck
            accepted = reward > 0
            if not accepted and temperature > 0:
                accepted = self._rng.random() < exp(reward / max(temperature, 1e-9))

            if accepted:
                current = candidate

            if current.throughput.bottleneck > best.throughput.bottleneck:
                best = current
                stable_count = 0
            else:
                stable_count += 1

            if self.config.keep_history:
                history.append(
                    {
                        "iteration": iteration,
                        "temperature": temperature,
                        "reward": reward,
                        "accepted": accepted,
                        "current_bottleneck": current.throughput.bottleneck,
                        "candidate_bottleneck": candidate.throughput.bottleneck,
                        "best_bottleneck": best.throughput.bottleneck,
                    }
                )

            if stable_count >= self.config.stability_iterations:
                break
            temperature *= self.config.cooling_rate

        best.iterations = iterations_run
        best.metadata["history"] = history
        best.metadata["capacity"] = dict(capacity)
        best.metadata["global_scheduler"] = self.config
        best.metadata["effective_block_size"] = self._effective_block_size()
        return best

    def _evaluate(
        self,
        workload: WorkloadProfile,
        allocation: AllocationMatrix,
        previous_parallelism=None,
    ) -> DeploymentPlan:
        previous_key = None
        if self.local_scheduler.config.cost_aware and previous_parallelism is not None:
            previous_key = tuple(
                (key, strategy.as_tuple())
                for key, strategy in sorted(previous_parallelism.items())
            )
        key = (
            allocation.as_key(),
            round(workload.arrival_rate, 12),
            workload.mean_input,
            workload.mean_output,
            workload.max_batch_size,
            workload.input_lengths,
            workload.output_lengths,
            previous_key,
        )
        if key not in self._cache:
            self._cache[key] = self.local_scheduler.evaluate(
                workload,
                allocation,
                previous_parallelism=previous_parallelism,
            )
        return self._cache[key]

    def _generate_candidate(
        self,
        allocation: AllocationMatrix,
        plan: DeploymentPlan,
        capacity: Mapping[str, int],
    ) -> AllocationMatrix:
        block_size = self._effective_block_size()
        for _ in range(32):
            src = self._sample_source(plan, allocation, capacity)
            dst = self._sample_destination(plan)
            if src == dst:
                continue
            if src is None:
                movable = [
                    hw for hw in capacity
                    if allocation.unused_for_hardware(capacity, hw) >= block_size
                ]
            else:
                movable = [
                    hw for hw in allocation.hardware_types()
                    if allocation.get(src, hw) >= block_size
                ]
            if not movable:
                continue
            hardware = self._rng.choice(movable)
            candidate = allocation.clone()
            if src is None and candidate.add(dst, hardware, block_size, capacity):
                return candidate
            if src is not None and candidate.move(src, dst, hardware, block_size):
                return candidate
        return allocation.clone()

    def _sample_destination(self, plan: DeploymentPlan) -> str:
        eps = 1e-9
        weights = [
            1.0 / max(plan.throughput.by_worker.get(worker, 0.0), eps)
            for worker in WORKER_TYPES
        ]
        return self._weighted_choice(list(WORKER_TYPES), weights)

    def _sample_source(
        self,
        plan: DeploymentPlan,
        allocation: AllocationMatrix,
        capacity: Mapping[str, int],
    ):
        items = list(WORKER_TYPES)
        weights = [
            max(plan.throughput.by_worker.get(worker, 0.0), 1e-9)
            for worker in WORKER_TYPES
        ]
        if self.config.allow_empty_source and allocation.total_unused(capacity) >= self._effective_block_size():
            items.append(None)
            average_weight = sum(weights) / max(len(weights), 1)
            weights.append(max(average_weight, plan.throughput.bottleneck, 1e-9))
        return self._weighted_choice(items, weights)

    def _effective_block_size(self) -> int:
        if self.config.block_size > 0:
            return self.config.block_size
        size = self.config.model_size_billions
        if size <= 13:
            return 1
        if size <= 34:
            return 2
        if size <= 90:
            return 4
        return 8

    def _weighted_choice(self, items: List[object], weights: List[float]):
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(items)
        point = self._rng.random() * total
        acc = 0.0
        for item, weight in zip(items, weights):
            acc += weight
            if point <= acc:
                return item
        return items[-1]
