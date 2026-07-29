"""Bounded stability validation for live autoscaling proposals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from simulator.scheduling import (
    AllocationMatrix,
    DeploymentPlan,
    HexGenSchedulingFramework,
    WORKER_TYPES,
    WorkloadProfile,
)


@dataclass(frozen=True)
class StabilityValidationConfig:
    max_rounds: int = 3
    max_candidates: int = 16
    timeout_s: float = 5.0


def stabilize_autoscaling_plan(
    framework: HexGenSchedulingFramework,
    workload: WorkloadProfile,
    current_plan: DeploymentPlan,
    proposed_plan: DeploymentPlan,
    capacity: Mapping[str, int],
    *,
    config: Optional[StabilityValidationConfig] = None,
    allow_scale_in: bool = True,
) -> Tuple[DeploymentPlan, Dict[str, object]]:
    """Resolve a proportional proposal that would immediately reverse or cycle."""

    validator = _StabilityValidator(
        framework=framework,
        workload=workload,
        current_plan=current_plan,
        capacity=capacity,
        config=config or StabilityValidationConfig(),
        allow_scale_in=allow_scale_in,
    )
    return validator.run(proposed_plan)


class _StabilityValidator:
    def __init__(
        self,
        *,
        framework: HexGenSchedulingFramework,
        workload: WorkloadProfile,
        current_plan: DeploymentPlan,
        capacity: Mapping[str, int],
        config: StabilityValidationConfig,
        allow_scale_in: bool,
    ):
        self.framework = framework
        self.workload = workload
        self.current_plan = current_plan
        self.capacity = capacity
        self.config = config
        self.allow_scale_in = allow_scale_in
        self.started_at = perf_counter()
        self.max_candidates = max(0, int(config.max_candidates))
        self.timeout_s = max(0.0, float(config.timeout_s))
        safe_utilization = max(
            1e-9,
            framework.autoscaling_config.target_utilization
            + framework.autoscaling_config.hysteresis,
        )
        self.required_throughput = workload.arrival_rate / safe_utilization
        self.evaluated: Dict[Tuple, DeploymentPlan] = {
            current_plan.allocation.as_key(): current_plan,
        }
        self.new_evaluations = 0
        self.timed_out = False

    def run(
        self,
        proposed_plan: DeploymentPlan,
    ) -> Tuple[DeploymentPlan, Dict[str, object]]:
        current_key = self.current_plan.allocation.as_key()
        proposed_key = proposed_plan.allocation.as_key()
        self.evaluated[proposed_key] = proposed_plan
        anchors = [
            self.current_plan.allocation.clone(),
            proposed_plan.allocation.clone(),
        ]
        proposal_path = [
            _allocation_values(self.current_plan.allocation),
            _allocation_values(proposed_plan.allocation),
        ]
        proposal_allowed = self._candidate_allowed(proposed_plan)
        triggered = not proposal_allowed or not self._is_safe(proposed_plan)
        fixed_point_keys = set()
        path_plan = proposed_plan
        seen = {proposed_key}

        for _ in range(max(1, int(self.config.max_rounds))):
            follow_up = self.framework.proportional_scale_allocation(
                self.workload,
                path_plan,
                self.capacity,
            )
            follow_up_key = follow_up.as_key()
            if follow_up_key == path_plan.allocation.as_key():
                fixed_point_keys.add(follow_up_key)
                break

            triggered = True
            anchors.append(follow_up.clone())
            proposal_path.append(_allocation_values(follow_up))
            if follow_up_key in seen:
                break
            seen.add(follow_up_key)
            next_plan = self._evaluate(follow_up)
            if next_plan is None:
                break
            path_plan = next_plan

        fixed_safe = [
            plan
            for key, plan in self.evaluated.items()
            if key in fixed_point_keys
            and self._is_safe(plan)
            and self._candidate_allowed(plan)
        ]
        if fixed_safe:
            selected = min(fixed_safe, key=self._rank)
            selection = "stable_fixed_point"
        else:
            if triggered:
                for allocation in self._local_allocations(anchors):
                    if allocation.as_key() in self.evaluated:
                        continue
                    if self._evaluate(allocation) is None:
                        break
            selected, selection = self._select_fallback()

        if not proposal_allowed and selected.allocation.as_key() == current_key:
            selection = "backlog_hold"
        return selected, self._metadata(
            triggered=triggered,
            selection=selection,
            proposal_path=proposal_path,
            selected=selected,
        )

    def _evaluate(self, allocation: AllocationMatrix) -> Optional[DeploymentPlan]:
        key = allocation.as_key()
        if key in self.evaluated:
            return self.evaluated[key]
        if not self._within_budget():
            return None
        plan = self.framework.evaluate_allocation(
            self.workload,
            allocation,
            previous_parallelism=self.current_plan.parallelism,
        )
        self.evaluated[key] = plan
        self.new_evaluations += 1
        if self.timeout_s > 0 and perf_counter() - self.started_at >= self.timeout_s:
            self.timed_out = True
        return plan

    def _within_budget(self) -> bool:
        if self.new_evaluations >= self.max_candidates:
            return False
        if self.timeout_s > 0 and perf_counter() - self.started_at >= self.timeout_s:
            self.timed_out = True
            return False
        return True

    def _select_fallback(self) -> Tuple[DeploymentPlan, str]:
        allowed = [plan for plan in self.evaluated.values() if self._candidate_allowed(plan)]
        if not allowed:
            allowed = [self.current_plan]
        safe = [plan for plan in allowed if self._is_safe(plan)]
        stable_safe = [
            plan
            for plan in safe
            if self.framework.proportional_scale_allocation(
                self.workload,
                plan,
                self.capacity,
            ).as_key()
            == plan.allocation.as_key()
        ]
        if stable_safe:
            return min(stable_safe, key=self._rank), "stable_fixed_point"
        if safe:
            return min(safe, key=self._rank), "quantization_hold"
        return (
            max(
                allowed,
                key=lambda plan: (
                    plan.throughput.bottleneck,
                    -plan.allocation.total_gpus(),
                    -plan.cost_per_hour,
                ),
            ),
            "capacity_limited",
        )

    def _candidate_allowed(self, candidate: DeploymentPlan) -> bool:
        if self.allow_scale_in:
            return True
        return all(
            _worker_total(candidate.allocation, worker)
            >= _worker_total(self.current_plan.allocation, worker)
            for worker in WORKER_TYPES
        )

    def _is_safe(self, plan: DeploymentPlan) -> bool:
        return plan.throughput.bottleneck + 1e-9 >= self.required_throughput

    def _rank(self, plan: DeploymentPlan) -> Tuple[float, ...]:
        reconfiguration_cost = sum(
            slice_plan.reconfiguration_cost_s
            for slice_plan in plan.slice_plans.values()
        )
        allocation_distance = sum(
            abs(
                plan.allocation.get(worker, hardware)
                - self.current_plan.allocation.get(worker, hardware)
            )
            for worker in WORKER_TYPES
            for hardware in set(plan.allocation.hardware_types())
            | set(self.current_plan.allocation.hardware_types())
        )
        total_gpus = float(plan.allocation.total_gpus())
        if self.framework.local_scheduler.config.cost_aware:
            return (
                plan.cost_per_hour,
                total_gpus,
                reconfiguration_cost,
                float(allocation_distance),
                -plan.throughput.bottleneck,
            )
        return (
            total_gpus,
            plan.cost_per_hour,
            reconfiguration_cost,
            float(allocation_distance),
            -plan.throughput.bottleneck,
        )

    def _local_allocations(
        self,
        anchors: Sequence[AllocationMatrix],
    ) -> List[AllocationMatrix]:
        unique_anchors = {anchor.as_key(): anchor for anchor in anchors}
        anchor_totals = [
            tuple(_worker_total(anchor, worker) for worker in WORKER_TYPES)
            for anchor in unique_anchors.values()
        ]
        ranges = self._worker_ranges(anchor_totals)
        total_capacity = sum(max(0, int(value)) for value in self.capacity.values())
        targets = [
            dict(zip(WORKER_TYPES, values))
            for values in product(*(ranges[worker] for worker in WORKER_TYPES))
            if sum(values) <= total_capacity
        ]
        targets.sort(
            key=lambda target: (
                min(
                    sum(
                        abs(target[worker] - totals[index])
                        for index, worker in enumerate(WORKER_TYPES)
                    )
                    for totals in anchor_totals
                ),
                sum(target.values()),
                tuple(target[worker] for worker in WORKER_TYPES),
            )
        )

        allocations = []
        seen = set(unique_anchors.keys())
        anchor_items = list(unique_anchors.values())
        for target in targets:
            seed = min(
                anchor_items,
                key=lambda anchor: sum(
                    abs(target[worker] - _worker_total(anchor, worker))
                    for worker in WORKER_TYPES
                ),
            )
            allocation = _rebuild_allocation(seed, self.capacity, target)
            key = allocation.as_key()
            if key in seen or any(
                _worker_total(allocation, worker) != target[worker]
                for worker in WORKER_TYPES
            ):
                continue
            seen.add(key)
            allocations.append(allocation)
        return allocations

    def _worker_ranges(
        self,
        anchor_totals: Sequence[Tuple[int, ...]],
    ) -> Dict[str, Tuple[int, ...]]:
        decode_choices = {
            int(choice)
            for choice in self.framework.autoscaling_config.decode_worker_gpu_choices
            if int(choice) > 0
        }
        ranges = {}
        for index, worker in enumerate(WORKER_TYPES):
            observed = {max(1, totals[index]) for totals in anchor_totals}
            low, high = min(observed), max(observed)
            if worker == "pre":
                options = set(range(low, high + 1))
            else:
                options = {choice for choice in decode_choices if low <= choice <= high}
                options.update(observed)
            ranges[worker] = tuple(sorted(options))
        return ranges

    def _metadata(
        self,
        *,
        triggered: bool,
        selection: str,
        proposal_path: Sequence[Mapping[str, Mapping[str, int]]],
        selected: DeploymentPlan,
    ) -> Dict[str, object]:
        return {
            "triggered": triggered,
            "selection": selection,
            "proposal_path": list(proposal_path),
            "required_safe_throughput_req_s": self.required_throughput,
            "evaluated_candidates": len(self.evaluated),
            "safe_candidates": sum(self._is_safe(plan) for plan in self.evaluated.values()),
            "elapsed_s": perf_counter() - self.started_at,
            "timed_out": self.timed_out,
            "allow_scale_in": self.allow_scale_in,
            "selected_allocation": _allocation_values(selected.allocation),
            "selected_throughput_req_s": selected.throughput.bottleneck,
        }


def _rebuild_allocation(
    seed: AllocationMatrix,
    capacity: Mapping[str, int],
    targets: Mapping[str, int],
) -> AllocationMatrix:
    rebuilt = AllocationMatrix.zeros(capacity.keys())
    for worker in WORKER_TYPES:
        remaining = max(0, int(targets.get(worker, 0)))
        hardware_order = sorted(
            capacity,
            key=lambda hardware: (-seed.get(worker, hardware), hardware),
        )
        while remaining > 0:
            added = False
            for hardware in hardware_order:
                if rebuilt.unused_for_hardware(capacity, hardware) <= 0:
                    continue
                rebuilt.add(worker, hardware, 1, capacity)
                remaining -= 1
                added = True
                break
            if not added:
                break
    return rebuilt


def _worker_total(allocation: AllocationMatrix, worker: str) -> int:
    return sum(allocation.values.get(worker, {}).values())


def _allocation_values(allocation: AllocationMatrix) -> Dict[str, Dict[str, int]]:
    return {
        worker: dict(allocation.values.get(worker, {}))
        for worker in WORKER_TYPES
    }
