"""High-level HexGen-3 scheduling framework orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Dict, List, Mapping, Optional, Tuple

from .estimator import SimulatorEstimator
from .global_scheduler import GlobalScheduler, GlobalSchedulerConfig
from .local_scheduler import LocalScheduler, LocalSchedulerConfig
from .types import (
    AllocationMatrix,
    DeploymentPlan,
    ParallelismStrategy,
    WORKER_TYPES,
    WorkloadProfile,
)


@dataclass
class AutoscalingConfig:
    target_utilization: float = 0.75
    hysteresis: float = 0.08
    min_scale_factor: float = 0.5
    max_scale_factor: float = 2.0
    decode_worker_gpu_choices: Tuple[int, ...] = (1, 2, 4, 8)
    global_search_after_scaling: bool = False


class HexGenSchedulingFramework:
    """Paper-style hierarchical scheduler for fully disaggregated serving."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        local_config: Optional[LocalSchedulerConfig] = None,
        global_config: Optional[GlobalSchedulerConfig] = None,
        autoscaling_config: Optional[AutoscalingConfig] = None,
        kv_transfer_bandwidth_gbps: float = 100.0,
        activation_bandwidth_gbps: float = 100.0,
        routing_policy: str = "random",
    ):
        self.estimator = SimulatorEstimator(
            model_id=model_id,
            kv_transfer_bandwidth_gbps=kv_transfer_bandwidth_gbps,
            activation_bandwidth_gbps=activation_bandwidth_gbps,
        )
        self.local_scheduler = LocalScheduler(self.estimator, local_config)
        self.global_scheduler = GlobalScheduler(self.local_scheduler, global_config)
        self.autoscaling_config = autoscaling_config or AutoscalingConfig()
        self.model_id = model_id
        self.kv_transfer_bandwidth_gbps = kv_transfer_bandwidth_gbps
        self.activation_bandwidth_gbps = activation_bandwidth_gbps
        self.routing_policy = routing_policy

    def optimize(
        self,
        workload: WorkloadProfile,
        capacity: Mapping[str, int],
        initial_allocation: Optional[AllocationMatrix] = None,
        previous_plan: Optional[DeploymentPlan] = None,
    ) -> DeploymentPlan:
        return self.global_scheduler.optimize(
            workload,
            capacity,
            initial_allocation,
            previous_plan=previous_plan,
        )

    def reschedule(
        self,
        workload: WorkloadProfile,
        previous_plan: DeploymentPlan,
        capacity: Mapping[str, int],
    ) -> DeploymentPlan:
        """Autoscaling-aware reschedule with proportional scaling and warm start."""

        scaled = self.proportional_scale_allocation(workload, previous_plan, capacity)
        if self.autoscaling_config.global_search_after_scaling:
            searched_plan = self.optimize(
                workload,
                capacity,
                initial_allocation=scaled,
                previous_plan=previous_plan,
            )
            plan = self._apply_autoscaling_allocation_constraints(
                workload,
                searched_plan,
                previous_plan,
                capacity,
            )
        else:
            plan = self.evaluate_allocation(
                workload,
                scaled,
                previous_parallelism=previous_plan.parallelism,
            )
        plan.metadata["autoscaling"] = {
            "worker_expansion": self.worker_expansion_factors(workload, previous_plan),
            "initial_scaled_allocation": scaled.values,
            "config": asdict(self.autoscaling_config),
        }
        return plan

    def worker_expansion_factors(
        self,
        workload: WorkloadProfile,
        previous_plan: DeploymentPlan,
    ) -> Dict[str, float]:
        config = self.autoscaling_config
        factors: Dict[str, float] = {}
        for worker in WORKER_TYPES:
            throughput = max(previous_plan.throughput.by_worker.get(worker, 0.0), 1e-9)
            utilization = workload.arrival_rate / throughput
            if utilization > config.target_utilization + config.hysteresis:
                factor = utilization / max(config.target_utilization, 1e-9)
            elif utilization < config.target_utilization - config.hysteresis:
                factor = utilization / max(config.target_utilization, 1e-9)
            else:
                factor = 1.0
            factors[worker] = min(config.max_scale_factor, max(config.min_scale_factor, factor))
        return factors

    def proportional_scale_allocation(
        self,
        workload: WorkloadProfile,
        previous_plan: DeploymentPlan,
        capacity: Mapping[str, int],
    ) -> AllocationMatrix:
        factors = self.worker_expansion_factors(workload, previous_plan)
        allocation = AllocationMatrix.zeros(capacity.keys())
        for worker in WORKER_TYPES:
            for hardware in capacity:
                old = previous_plan.allocation.get(worker, hardware)
                if old <= 0:
                    continue
                allocation.set(worker, hardware, ceil(old * factors[worker]))

        self._fit_allocation_to_capacity(allocation, capacity, factors)
        self._ensure_non_empty_workers(allocation, capacity)
        return self._quantize_decode_worker_allocations(allocation, capacity)

    def _apply_autoscaling_allocation_constraints(
        self,
        workload: WorkloadProfile,
        plan: DeploymentPlan,
        previous_plan: DeploymentPlan,
        capacity: Mapping[str, int],
    ) -> DeploymentPlan:
        constrained = self._quantize_decode_worker_allocations(plan.allocation, capacity)
        if constrained.as_key() == plan.allocation.as_key():
            return plan

        constrained_plan = self.evaluate_allocation(
            workload,
            constrained,
            previous_parallelism=previous_plan.parallelism,
        )
        constrained_plan.iterations = plan.iterations
        constrained_plan.metadata.update(plan.metadata)
        constrained_plan.metadata["unconstrained_search_allocation"] = plan.allocation.values
        return constrained_plan

    def _quantize_decode_worker_allocations(
        self,
        allocation: AllocationMatrix,
        capacity: Mapping[str, int],
    ) -> AllocationMatrix:
        choices = tuple(
            sorted(
                {
                    int(choice)
                    for choice in self.autoscaling_config.decode_worker_gpu_choices
                    if int(choice) > 0
                }
            )
        )
        if not choices:
            return allocation.clone()

        total_capacity = sum(max(0, int(count)) for count in capacity.values())
        valid_pairs = [
            (attn, ffn)
            for attn in choices
            for ffn in choices
            if attn + ffn <= total_capacity - 1
        ]
        if not valid_pairs:
            return allocation.clone()

        desired = {
            worker: max(1, self._worker_total(allocation, worker))
            for worker in WORKER_TYPES
        }
        target_attn, target_ffn = min(
            valid_pairs,
            key=lambda pair: (
                abs(pair[0] - desired["attn"]) + abs(pair[1] - desired["ffn"]),
                abs((pair[0] + pair[1]) - (desired["attn"] + desired["ffn"])),
                -min(pair),
            ),
        )
        target_pre = min(
            max(1, desired["pre"]),
            total_capacity - target_attn - target_ffn,
        )
        targets = {"pre": target_pre, "attn": target_attn, "ffn": target_ffn}
        return self._rebuild_allocation_with_worker_totals(allocation, capacity, targets)

    def _rebuild_allocation_with_worker_totals(
        self,
        allocation: AllocationMatrix,
        capacity: Mapping[str, int],
        targets: Mapping[str, int],
    ) -> AllocationMatrix:
        rebuilt = AllocationMatrix.zeros(capacity.keys())
        for worker in WORKER_TYPES:
            remaining = max(0, int(targets.get(worker, 0)))
            hardware_order = sorted(
                capacity,
                key=lambda hardware: (-allocation.get(worker, hardware), hardware),
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

    @staticmethod
    def _worker_total(allocation: AllocationMatrix, worker: str) -> int:
        return sum(allocation.get(worker, hw) for hw in allocation.hardware_types())

    def _fit_allocation_to_capacity(
        self,
        allocation: AllocationMatrix,
        capacity: Mapping[str, int],
        factors: Mapping[str, float],
    ) -> None:
        for hardware, total in capacity.items():
            while allocation.total_for_hardware(hardware) > total:
                candidates = [
                    worker for worker in WORKER_TYPES
                    if allocation.get(worker, hardware) > 0
                ]
                if not candidates:
                    break
                worker = max(
                    candidates,
                    key=lambda item: (allocation.get(item, hardware), -factors.get(item, 1.0)),
                )
                allocation.set(worker, hardware, allocation.get(worker, hardware) - 1)

    def _ensure_non_empty_workers(self, allocation: AllocationMatrix, capacity: Mapping[str, int]) -> None:
        for worker in WORKER_TYPES:
            if sum(allocation.get(worker, hw) for hw in capacity) > 0:
                continue
            hardware = self._first_hardware_with_room(allocation, capacity)
            if hardware is not None:
                allocation.add(worker, hardware, 1, capacity)

    @staticmethod
    def _first_hardware_with_room(allocation: AllocationMatrix, capacity: Mapping[str, int]) -> Optional[str]:
        for hardware in capacity:
            if allocation.unused_for_hardware(capacity, hardware) > 0:
                return hardware
        return None

    def evaluate_allocation(
        self,
        workload: WorkloadProfile,
        allocation: AllocationMatrix,
        previous_parallelism: Optional[Mapping[Tuple[str, str], ParallelismStrategy]] = None,
    ) -> DeploymentPlan:
        return self.local_scheduler.evaluate(workload, allocation, previous_parallelism)

    def build_afd_cluster_config(self, plan: DeploymentPlan):
        """Convert a plan to the copied simulator's AFD cluster configuration."""

        from simulator.core.cluster_manager import ClusterConfiguration, NodeConfiguration
        from simulator.core.config import ParallelConfig

        nodes: List[NodeConfiguration] = []
        for (worker, hardware), strategy in sorted(plan.parallelism.items()):
            for idx, replica in enumerate(strategy.replicas):
                if worker == "pre":
                    nodes.append(
                        NodeConfiguration(
                            node_id=f"{hardware}_pre_{idx}",
                            model_id=self.model_id,
                            hardware=hardware,
                            max_batch_size=1,
                            parallel_config=ParallelConfig(tensor_parallel_size=replica.tp),
                            pd_separation=True,
                            pd_prefill_only=True,
                            kv_transfer_bandwidth_gbps=self.kv_transfer_bandwidth_gbps,
                        )
                    )
                elif worker == "attn":
                    nodes.append(
                        NodeConfiguration(
                            node_id=f"{hardware}_attn_{idx}",
                            model_id=self.model_id,
                            hardware=hardware,
                            max_batch_size=max(1, replica.tp * 4),
                            parallel_config=ParallelConfig(tensor_parallel_size=replica.tp),
                            pd_separation=True,
                            pd_decode_only=True,
                            kv_transfer_bandwidth_gbps=self.kv_transfer_bandwidth_gbps,
                            afd_attention=True,
                            afd_enabled=True,
                        )
                    )
                elif worker == "ffn":
                    nodes.append(
                        NodeConfiguration(
                            node_id=f"{hardware}_ffn_{idx}",
                            model_id=self.model_id,
                            hardware=hardware,
                            max_batch_size=max(1, replica.tp * 4),
                            parallel_config=ParallelConfig(tensor_parallel_size=replica.tp),
                            pd_separation=True,
                            pd_decode_only=True,
                            kv_transfer_bandwidth_gbps=self.kv_transfer_bandwidth_gbps,
                            afd_ffn=True,
                            afd_enabled=True,
                        )
                    )

        return ClusterConfiguration(
            cluster_id="hexgen3_scheduled_afd",
            nodes=nodes,
            scheduler_algorithm=self.routing_policy,
            afd_enabled=True,
            afd_attention_batch_size=max(1, self._max_worker_batch(plan, "attn")),
            afd_ffn_max_batch_size=max(1, self._max_worker_batch(plan, "ffn")),
            afd_activation_bandwidth_gbps=self.activation_bandwidth_gbps,
        )

    def build_pd_cluster_config(self, plan: DeploymentPlan):
        """Build a P-D config by mapping pre slices to prefill and attn+ffn to decode."""

        from simulator.core.cluster_manager import ClusterConfiguration, NodeConfiguration
        from simulator.core.config import ParallelConfig

        nodes: List[NodeConfiguration] = []
        for (worker, hardware), strategy in sorted(plan.parallelism.items()):
            if worker == "pre":
                for idx, replica in enumerate(strategy.replicas):
                    nodes.append(
                        NodeConfiguration(
                            node_id=f"{hardware}_prefill_{idx}",
                            model_id=self.model_id,
                            hardware=hardware,
                            parallel_config=ParallelConfig(tensor_parallel_size=replica.tp),
                            pd_separation=True,
                            pd_prefill_only=True,
                            kv_transfer_bandwidth_gbps=self.kv_transfer_bandwidth_gbps,
                        )
                    )
            elif worker in {"attn", "ffn"}:
                for idx, replica in enumerate(strategy.replicas):
                    nodes.append(
                        NodeConfiguration(
                            node_id=f"{hardware}_decode_{worker}_{idx}",
                            model_id=self.model_id,
                            hardware=hardware,
                            max_batch_size=max(1, replica.tp * 4),
                            parallel_config=ParallelConfig(tensor_parallel_size=replica.tp),
                            pd_separation=True,
                            pd_decode_only=True,
                            kv_transfer_bandwidth_gbps=self.kv_transfer_bandwidth_gbps,
                        )
                    )
        return ClusterConfiguration(
            cluster_id="hexgen3_scheduled_pd",
            nodes=nodes,
            scheduler_algorithm=self.routing_policy,
        )

    @staticmethod
    def _max_worker_batch(plan: DeploymentPlan, worker_type: str) -> int:
        batches = [
            max(1, replica.tp * 4)
            for (worker, _), strategy in plan.parallelism.items()
            if worker == worker_type
            for replica in strategy.replicas
        ]
        return max(batches) if batches else 1


def plan_to_dict(plan: DeploymentPlan) -> Dict[str, object]:
    parallelism = {
        f"{worker}/{hardware}": strategy.as_dict()
        for (worker, hardware), strategy in sorted(plan.parallelism.items())
    }
    allocation = {
        worker: dict(plan.allocation.values.get(worker, {}))
        for worker in WORKER_TYPES
    }
    slices = {
        f"{worker}/{hardware}": {
            "gpus": slice_plan.gpus,
            "throughput_req_s": slice_plan.throughput,
            "reconfiguration_cost_s": slice_plan.reconfiguration_cost_s,
            "score": slice_plan.score,
            "strategy": (
                slice_plan.strategy.as_dict()
                if slice_plan.strategy is not None
                else None
            ),
        }
        for (worker, hardware), slice_plan in sorted(plan.slice_plans.items())
    }
    return {
        "allocation": allocation,
        "parallelism": parallelism,
        "slice_plans": slices,
        "throughput_req_s": dict(plan.throughput.by_worker),
        "system_throughput_req_s": plan.throughput.bottleneck,
        "estimated_latency_s": plan.estimated_latency_s,
        "tail_latency_s": dict(plan.tail_latency_s),
        "cost_per_hour": plan.cost_per_hour,
        "req_per_dollar": plan.req_per_dollar,
        "iterations": plan.iterations,
        "metadata": {
            k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
            for k, v in plan.metadata.items()
            if k != "history"
        },
        "history": plan.metadata.get("history", []),
    }
