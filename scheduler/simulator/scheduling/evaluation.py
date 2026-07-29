"""Focused evaluation helpers for scheduler and autoscaler unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .framework import HexGenSchedulingFramework, plan_to_dict
from .types import AllocationMatrix, DeploymentPlan, WORKER_TYPES, WorkloadProfile


FrameworkFactory = Callable[[], HexGenSchedulingFramework]


@dataclass(frozen=True)
class SchedulingCase:
    """One cluster size/configuration to schedule."""

    name: str
    capacity: Mapping[str, int]


@dataclass(frozen=True)
class SchedulingMeasurement:
    """Timing and outcome for one scheduling run."""

    case_name: str
    repeat: int
    capacity: Dict[str, int]
    elapsed_s: float
    iterations: int
    system_throughput_req_s: float
    estimated_latency_s: float
    cost_per_hour: float
    req_per_dollar: float
    allocation: Dict[str, Dict[str, int]]
    parallelism: Dict[str, object]
    throughput_req_s: Dict[str, float]


@dataclass(frozen=True)
class LoadWindow:
    """One autoscaling window with a different duration or request load."""

    name: str
    start_s: float
    duration_s: float
    arrival_rate: float
    input_scale: float = 1.0
    output_scale: float = 1.0
    capacity: Optional[Mapping[str, int]] = None

    def workload_from(self, base: WorkloadProfile) -> WorkloadProfile:
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        return WorkloadProfile(
            arrival_rate=self.arrival_rate,
            input_lengths=tuple(_scale_tokens(base.input_lengths, self.input_scale)),
            output_lengths=tuple(_scale_tokens(base.output_lengths, self.output_scale)),
            max_batch_size=base.max_batch_size,
        )


@dataclass(frozen=True)
class AutoscalingMeasurement:
    """Autoscaling decision and final schedule for one load window."""

    window_name: str
    start_s: float
    duration_s: float
    arrival_rate: float
    capacity: Dict[str, int]
    elapsed_s: float
    iterations: int
    worker_expansion: Dict[str, float]
    action_by_worker: Dict[str, str]
    allocation_delta_by_worker: Dict[str, int]
    initial_scaled_allocation: Dict[str, Dict[str, int]]
    allocation: Dict[str, Dict[str, int]]
    parallelism: Dict[str, object]
    throughput_req_s: Dict[str, float]
    system_throughput_req_s: float
    estimated_latency_s: float
    tail_latency_s: Dict[str, float]
    reconfiguration_cost_s: float


def measure_scheduling_cases(
    framework_factory: FrameworkFactory,
    workload: WorkloadProfile,
    cases: Sequence[SchedulingCase],
    repeats: int = 1,
) -> List[SchedulingMeasurement]:
    """Run the global scheduler for several cluster sizes and capture timings."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    measurements: List[SchedulingMeasurement] = []
    for case in cases:
        capacity = _normalize_capacity(case.capacity)
        for repeat in range(repeats):
            framework = framework_factory()
            start = perf_counter()
            plan = framework.optimize(workload, capacity)
            elapsed = perf_counter() - start
            measurements.append(
                _scheduling_measurement(
                    case_name=case.name,
                    repeat=repeat,
                    capacity=capacity,
                    elapsed_s=elapsed,
                    plan=plan,
                )
            )
    return measurements


def simulate_autoscaling_windows(
    framework: HexGenSchedulingFramework,
    base_workload: WorkloadProfile,
    initial_capacity: Mapping[str, int],
    windows: Sequence[LoadWindow],
    initial_plan: Optional[DeploymentPlan] = None,
    use_window_duration_for_stability: bool = True,
) -> List[AutoscalingMeasurement]:
    """Reschedule a deployment over variable-load windows.

    The first window creates or records the starting plan. Later windows call
    ``HexGenSchedulingFramework.reschedule`` so tests can inspect proportional
    autoscaling factors, warm-start allocation, and the final schedule.
    """

    if not windows:
        raise ValueError("windows must not be empty")

    previous_plan = initial_plan
    previous_allocation: Optional[AllocationMatrix] = (
        initial_plan.allocation.clone() if initial_plan is not None else None
    )
    measurements: List[AutoscalingMeasurement] = []

    for index, window in enumerate(windows):
        workload = window.workload_from(base_workload)
        capacity = _normalize_capacity(window.capacity or initial_capacity)
        if use_window_duration_for_stability:
            framework.local_scheduler.config.stability_window_s = window.duration_s

        start = perf_counter()
        if previous_plan is None:
            plan = framework.optimize(workload, capacity)
            constrained = framework._quantize_decode_worker_allocations(plan.allocation, capacity)
            if constrained.as_key() != plan.allocation.as_key():
                constrained_plan = framework.evaluate_allocation(workload, constrained)
                constrained_plan.iterations = plan.iterations
                constrained_plan.metadata.update(plan.metadata)
                constrained_plan.metadata["autoscaling_initial_unconstrained_allocation"] = (
                    plan.allocation.values
                )
                plan = constrained_plan
        else:
            plan = framework.reschedule(workload, previous_plan, capacity)
        elapsed = perf_counter() - start

        measurements.append(
            _autoscaling_measurement(
                window=window,
                capacity=capacity,
                elapsed_s=elapsed,
                plan=plan,
                previous_allocation=previous_allocation,
                initial_window=index == 0 and initial_plan is None,
            )
        )
        previous_plan = plan
        previous_allocation = plan.allocation.clone()

    return measurements


def _scheduling_measurement(
    case_name: str,
    repeat: int,
    capacity: Mapping[str, int],
    elapsed_s: float,
    plan: DeploymentPlan,
) -> SchedulingMeasurement:
    plan_dict = plan_to_dict(plan)
    return SchedulingMeasurement(
        case_name=case_name,
        repeat=repeat,
        capacity=dict(capacity),
        elapsed_s=elapsed_s,
        iterations=plan.iterations,
        system_throughput_req_s=float(plan_dict["system_throughput_req_s"]),
        estimated_latency_s=float(plan_dict["estimated_latency_s"]),
        cost_per_hour=float(plan_dict["cost_per_hour"]),
        req_per_dollar=float(plan_dict["req_per_dollar"]),
        allocation=plan_dict["allocation"],
        parallelism=plan_dict["parallelism"],
        throughput_req_s=plan_dict["throughput_req_s"],
    )


def _autoscaling_measurement(
    window: LoadWindow,
    capacity: Mapping[str, int],
    elapsed_s: float,
    plan: DeploymentPlan,
    previous_allocation: Optional[AllocationMatrix],
    initial_window: bool,
) -> AutoscalingMeasurement:
    plan_dict = plan_to_dict(plan)
    autoscaling = plan.metadata.get("autoscaling", {})
    if initial_window:
        worker_expansion = {worker: 1.0 for worker in WORKER_TYPES}
        action_by_worker = {worker: "initial" for worker in WORKER_TYPES}
        initial_scaled_allocation = plan_dict["allocation"]
    else:
        worker_expansion = {
            worker: float(value)
            for worker, value in autoscaling.get("worker_expansion", {}).items()
        }
        action_by_worker = {
            worker: _action_for_factor(worker_expansion.get(worker, 1.0))
            for worker in WORKER_TYPES
        }
        initial_scaled_allocation = autoscaling.get(
            "initial_scaled_allocation",
            plan_dict["allocation"],
        )

    return AutoscalingMeasurement(
        window_name=window.name,
        start_s=window.start_s,
        duration_s=window.duration_s,
        arrival_rate=window.arrival_rate,
        capacity=dict(capacity),
        elapsed_s=elapsed_s,
        iterations=plan.iterations,
        worker_expansion=worker_expansion,
        action_by_worker=action_by_worker,
        allocation_delta_by_worker=_allocation_delta(previous_allocation, plan.allocation),
        initial_scaled_allocation=initial_scaled_allocation,
        allocation=plan_dict["allocation"],
        parallelism=plan_dict["parallelism"],
        throughput_req_s=plan_dict["throughput_req_s"],
        system_throughput_req_s=float(plan_dict["system_throughput_req_s"]),
        estimated_latency_s=float(plan_dict["estimated_latency_s"]),
        tail_latency_s=plan_dict["tail_latency_s"],
        reconfiguration_cost_s=sum(
            slice_plan.reconfiguration_cost_s
            for slice_plan in plan.slice_plans.values()
        ),
    )


def _allocation_delta(
    previous: Optional[AllocationMatrix],
    current: AllocationMatrix,
) -> Dict[str, int]:
    if previous is None:
        return {worker: current_total(current, worker) for worker in WORKER_TYPES}
    return {
        worker: current_total(current, worker) - current_total(previous, worker)
        for worker in WORKER_TYPES
    }


def current_total(allocation: AllocationMatrix, worker: str) -> int:
    return sum(allocation.get(worker, hardware) for hardware in allocation.hardware_types())


def _action_for_factor(factor: float) -> str:
    if factor > 1.0:
        return "scale_up"
    if factor < 1.0:
        return "scale_down"
    return "hold"


def _scale_tokens(values: Sequence[int], scale: float) -> List[int]:
    if scale <= 0:
        raise ValueError("token scale must be positive")
    return [max(1, int(round(value * scale))) for value in values]


def _normalize_capacity(capacity: Mapping[str, int]) -> Dict[str, int]:
    normalized = {hardware: int(count) for hardware, count in capacity.items()}
    if any(count < 0 for count in normalized.values()):
        raise ValueError("capacity values must be non-negative")
    if sum(normalized.values()) < len(WORKER_TYPES):
        raise ValueError("capacity must have at least one GPU per worker type")
    return normalized
