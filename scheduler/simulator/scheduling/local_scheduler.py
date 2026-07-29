"""Local constrained parallelism search for a fixed allocation matrix."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .estimator import SimulatorEstimator
from .types import (
    AllocationMatrix,
    DeploymentPlan,
    ParallelismStrategy,
    ParallelismReplica,
    SlicePlan,
    ThroughputProfile,
    WORKER_TYPES,
    WorkloadProfile,
)


@dataclass
class LocalSchedulerConfig:
    node_gpus: int = 8
    enumerate_non_uniform: bool = True
    max_local_strategies: int = 4096
    enable_expert_parallel: bool = False
    num_experts: int = 1
    match_attn_ffn_dp: bool = True
    cost_aware: bool = False
    stability_window_s: float = 300.0
    reload_bandwidth_gbps: float = 600.0
    model_size_gb: float = 16.0


class LocalScheduler:
    """Enumerates valid p=<DP,TP,EP> strategies for each worker/hardware slice.

    Unlike the first prototype, this enumerates non-uniform DP replicas: for a
    slice with ``g`` allocated GPUs, every candidate satisfies
    ``sum(replica.tp) == g``. This matches the paper's local-search constraint
    C1 and still supports the old uniform strategy form.
    """

    def __init__(self, estimator: SimulatorEstimator, config: Optional[LocalSchedulerConfig] = None):
        self.estimator = estimator
        self.config = config or LocalSchedulerConfig()

    def enumerate_strategies(self, gpus: int) -> List[ParallelismStrategy]:
        if gpus <= 0:
            return []

        strategies: List[ParallelismStrategy] = []
        if self.config.enumerate_non_uniform:
            partitions = self._tp_partitions(
                remaining=gpus,
                min_tp=1,
                max_tp=min(gpus, self.config.node_gpus),
            )
        else:
            partitions = (
                (tp,) * (gpus // tp)
                for tp in self._divisors(gpus)
                if tp <= self.config.node_gpus
            )

        seen = set()
        for tp_partition in partitions:
            for strategy in self._strategies_for_tp_partition(tp_partition):
                if not self._is_strategy_valid(strategy, gpus):
                    continue
                key = strategy.as_tuple()
                if key in seen:
                    continue
                seen.add(key)
                strategies.append(strategy)
                if len(strategies) >= self.config.max_local_strategies:
                    return strategies
        return strategies

    def _tp_partitions(
        self,
        remaining: int,
        min_tp: int,
        max_tp: int,
        prefix: Tuple[int, ...] = (),
    ) -> Iterator[Tuple[int, ...]]:
        if remaining == 0:
            yield prefix
            return
        upper = min(max_tp, remaining)
        for tp in range(min_tp, upper + 1):
            yield from self._tp_partitions(
                remaining=remaining - tp,
                min_tp=tp,
                max_tp=max_tp,
                prefix=prefix + (tp,),
            )

    @staticmethod
    def _divisors(value: int) -> Iterable[int]:
        for candidate in range(1, value + 1):
            if value % candidate == 0:
                yield candidate

    def _strategies_for_tp_partition(self, tp_partition: Sequence[int]) -> Iterator[ParallelismStrategy]:
        if not self.config.enable_expert_parallel or self.config.num_experts <= 1:
            yield ParallelismStrategy.from_replicas(
                ParallelismReplica(tp=tp, ep=1) for tp in tp_partition
            )
            return

        intra_ep_choices = [self._intra_replica_ep_candidates(tp) for tp in tp_partition]
        for ep_partition in product(*intra_ep_choices):
            yield ParallelismStrategy.from_replicas(
                ParallelismReplica(tp=tp, ep=ep)
                for tp, ep in zip(tp_partition, ep_partition)
            )

        # Cross-replica EP is only legal when TP is identical across replicas.
        if len(set(tp_partition)) == 1:
            tp = tp_partition[0]
            for ep in self._cross_replica_ep_candidates(tp):
                yield ParallelismStrategy.from_replicas(
                    ParallelismReplica(tp=tp, ep=ep) for tp in tp_partition
                )

    def _intra_replica_ep_candidates(self, tp: int) -> List[int]:
        eps = []
        for ep in self._divisors(min(tp, self.config.num_experts)):
            if ep <= tp and tp % ep == 0 and self.config.num_experts % ep == 0:
                eps.append(ep)
        return eps or [1]

    def _cross_replica_ep_candidates(self, tp: int) -> List[int]:
        eps = []
        for ep in range(tp + 1, min(self.config.node_gpus, self.config.num_experts) + 1):
            if ep % tp == 0 and self.config.num_experts % ep == 0:
                eps.append(ep)
        return eps

    def _is_strategy_valid(self, strategy: ParallelismStrategy, gpus: int) -> bool:
        if strategy.gpus != gpus:
            return False
        has_cross_replica_ep = any(replica.ep > replica.tp for replica in strategy.replicas)
        if has_cross_replica_ep:
            if not all(replica.tp == strategy.replicas[0].tp for replica in strategy.replicas):
                return False
            if not all(replica.ep == strategy.replicas[0].ep for replica in strategy.replicas):
                return False
        for replica in strategy.replicas:
            if max(replica.tp, replica.ep) > self.config.node_gpus:
                return False
            if replica.ep <= replica.tp:
                if replica.tp % replica.ep != 0:
                    return False
            else:
                if replica.ep % replica.tp != 0:
                    return False
        return True

    def project_strategy(
        self,
        previous: Optional[ParallelismStrategy],
        gpus: int,
    ) -> Optional[ParallelismStrategy]:
        """Warm-start projection that preserves TP/EP and scales DP when possible."""

        if previous is None or gpus <= 0:
            return None
        if previous.gpus == gpus and self._is_strategy_valid(previous, gpus):
            return previous

        if previous.is_uniform:
            replica = previous.replicas[0]
            if gpus % replica.tp != 0:
                return None
            candidate = ParallelismStrategy(
                dp=gpus // replica.tp,
                tp=replica.tp,
                ep=replica.ep,
            )
            return candidate if self._is_strategy_valid(candidate, gpus) else None

        pattern = previous.replicas
        pattern_gpus = previous.gpus
        if pattern_gpus <= 0 or gpus % pattern_gpus != 0:
            return None
        candidate = ParallelismStrategy.from_replicas(pattern * (gpus // pattern_gpus))
        return candidate if self._is_strategy_valid(candidate, gpus) else None

    def _reconfiguration_cost_s(
        self,
        previous: Optional[ParallelismStrategy],
        candidate: ParallelismStrategy,
    ) -> float:
        if previous is None:
            return 0.0
        if previous.as_tuple() == candidate.as_tuple():
            return 0.0
        previous_kinds = {replica.as_tuple() for replica in previous.replicas}
        candidate_kinds = {replica.as_tuple() for replica in candidate.replicas}
        model_bytes = self.config.model_size_gb * 1e9
        bandwidth_bytes = max(self.config.reload_bandwidth_gbps, 1e-6) * 1e9
        base_reload = model_bytes / bandwidth_bytes
        if candidate_kinds.issubset(previous_kinds):
            added_replicas = max(0, candidate.dp - previous.dp)
            if added_replicas == 0:
                return 0.0
            return base_reload * (added_replicas / max(candidate.dp, 1))
        return base_reload

    def _candidate_score(
        self,
        throughput: float,
        previous_throughput: float,
        previous: Optional[ParallelismStrategy],
        candidate: ParallelismStrategy,
    ) -> Tuple[float, float]:
        if not self.config.cost_aware:
            return throughput, 0.0
        tau = self._reconfiguration_cost_s(previous, candidate)
        window = self.config.stability_window_s
        if previous is None:
            return throughput * window, tau
        if tau <= 0:
            return throughput * window, tau
        if tau >= window:
            return -1.0, tau

        # Paper-style amortization: reload if the throughput gain recovered
        # over the remaining stable window exceeds the service lost to reload.
        gain = (throughput - previous_throughput) * (window - tau)
        reload_loss = previous_throughput * tau
        if gain <= reload_loss:
            return -1.0, tau
        return throughput * (window - tau), tau

    def choose_strategy(
        self,
        worker_type: str,
        hardware: str,
        gpus: int,
        workload: WorkloadProfile,
        previous: Optional[ParallelismStrategy] = None,
    ) -> SlicePlan:
        best_strategy: Optional[ParallelismStrategy] = None
        best_throughput = 0.0
        best_tau = 0.0
        best_score = -1.0
        for plan in self._candidate_slice_plans(
            worker_type,
            hardware,
            gpus,
            workload,
            previous,
        ):
            if plan.score > best_score:
                best_score = plan.score
                best_tau = plan.reconfiguration_cost_s
                best_throughput = plan.throughput
                best_strategy = plan.strategy

        return SlicePlan(
            worker_type=worker_type,
            hardware=hardware,
            gpus=gpus,
            strategy=best_strategy,
            throughput=best_throughput,
            reconfiguration_cost_s=best_tau,
            score=best_score,
        )

    def _candidate_slice_plans(
        self,
        worker_type: str,
        hardware: str,
        gpus: int,
        workload: WorkloadProfile,
        previous: Optional[ParallelismStrategy] = None,
    ) -> List[SlicePlan]:
        projected_previous = self.project_strategy(previous, gpus)
        previous_throughput = 0.0
        if projected_previous is not None:
            previous_throughput = self.estimator.estimate_slice_throughput(
                worker_type=worker_type,
                hardware=hardware,
                strategy=projected_previous,
                workload=workload,
            )

        strategies = self.enumerate_strategies(gpus)
        if projected_previous is not None and projected_previous.as_tuple() not in {
            strategy.as_tuple() for strategy in strategies
        }:
            strategies.append(projected_previous)

        plans: List[SlicePlan] = []
        for strategy in strategies:
            throughput = self.estimator.estimate_slice_throughput(
                worker_type=worker_type,
                hardware=hardware,
                strategy=strategy,
                workload=workload,
            )
            score, tau = self._candidate_score(
                throughput,
                previous_throughput,
                projected_previous,
                strategy,
            )
            plans.append(
                SlicePlan(
                    worker_type=worker_type,
                    hardware=hardware,
                    gpus=gpus,
                    strategy=strategy,
                    throughput=throughput,
                    reconfiguration_cost_s=tau,
                    score=score,
                )
            )
        return plans

    def _choose_decode_strategies_with_matched_dp(
        self,
        workload: WorkloadProfile,
        allocation: AllocationMatrix,
        previous_parallelism: Optional[Mapping[Tuple[str, str], ParallelismStrategy]],
    ) -> Dict[Tuple[str, str], SlicePlan]:
        worker_options = {}
        for worker in ("attn", "ffn"):
            slices = []
            for hardware in allocation.hardware_types():
                gpus = allocation.get(worker, hardware)
                if gpus <= 0:
                    continue
                previous = None
                if previous_parallelism is not None:
                    previous = previous_parallelism.get((worker, hardware))
                candidates = self._candidate_slice_plans(
                    worker,
                    hardware,
                    gpus,
                    workload,
                    previous,
                )
                if candidates:
                    slices.append(((worker, hardware), candidates))
            worker_options[worker] = self._best_worker_combinations_by_dp(slices)

        if not worker_options["attn"] or not worker_options["ffn"]:
            return {}

        common_dp = set(worker_options["attn"]) & set(worker_options["ffn"])
        if not common_dp:
            return {}

        target_dp = max(
            common_dp,
            key=lambda dp: (
                min(
                    worker_options["attn"][dp]["throughput"],
                    worker_options["ffn"][dp]["throughput"],
                ),
                worker_options["attn"][dp]["score"] + worker_options["ffn"][dp]["score"],
                -abs(
                    worker_options["attn"][dp]["throughput"]
                    - worker_options["ffn"][dp]["throughput"]
                ),
            ),
        )
        selected = {}
        selected.update(worker_options["attn"][target_dp]["plans"])
        selected.update(worker_options["ffn"][target_dp]["plans"])
        return selected

    @staticmethod
    def _best_worker_combinations_by_dp(slices):
        states = {0: {"score": 0.0, "throughput": 0.0, "plans": {}}}
        for key, candidates in slices:
            next_states = {}
            for dp_sum, state in states.items():
                for candidate in candidates:
                    if candidate.strategy is None:
                        continue
                    next_dp = dp_sum + candidate.strategy.dp
                    score = state["score"] + candidate.score
                    throughput = state["throughput"] + candidate.throughput
                    plans = dict(state["plans"])
                    plans[key] = candidate
                    current = next_states.get(next_dp)
                    if current is None or (score, throughput) > (
                        current["score"],
                        current["throughput"],
                    ):
                        next_states[next_dp] = {
                            "score": score,
                            "throughput": throughput,
                            "plans": plans,
                        }
            states = next_states
        return {dp: state for dp, state in states.items() if dp > 0}

    def evaluate(
        self,
        workload: WorkloadProfile,
        allocation: AllocationMatrix,
        previous_parallelism: Optional[Mapping[Tuple[str, str], ParallelismStrategy]] = None,
    ) -> DeploymentPlan:
        slice_plans: Dict[Tuple[str, str], SlicePlan] = {}
        parallelism: Dict[Tuple[str, str], ParallelismStrategy] = {}
        worker_throughput = {worker: 0.0 for worker in WORKER_TYPES}

        independent_workers = ("pre",) if self.config.match_attn_ffn_dp else WORKER_TYPES
        for worker in independent_workers:
            for hardware in allocation.hardware_types():
                gpus = allocation.get(worker, hardware)
                if gpus <= 0:
                    continue
                previous = None
                if previous_parallelism is not None:
                    previous = previous_parallelism.get((worker, hardware))
                plan = self.choose_strategy(worker, hardware, gpus, workload, previous)
                slice_plans[(worker, hardware)] = plan
                if plan.strategy is not None:
                    parallelism[(worker, hardware)] = plan.strategy
                worker_throughput[worker] += plan.throughput

        if self.config.match_attn_ffn_dp:
            decode_plans = self._choose_decode_strategies_with_matched_dp(
                workload,
                allocation,
                previous_parallelism,
            )
            for (worker, hardware), plan in decode_plans.items():
                slice_plans[(worker, hardware)] = plan
                if plan.strategy is not None:
                    parallelism[(worker, hardware)] = plan.strategy
                worker_throughput[worker] += plan.throughput

        throughput = ThroughputProfile(worker_throughput)
        latency = self.estimator.estimate_latency(workload, throughput)
        tail_latency = self.estimator.estimate_tail_latency(workload, throughput)
        cost = self.estimator.estimate_cost_per_hour(allocation)
        req_per_dollar = throughput.bottleneck / (cost / 3600.0) if cost > 0 else 0.0
        return DeploymentPlan(
            allocation=allocation.clone(),
            parallelism=parallelism,
            slice_plans=slice_plans,
            throughput=throughput,
            estimated_latency_s=latency,
            cost_per_hour=cost,
            req_per_dollar=req_per_dollar,
            tail_latency_s=tail_latency,
        )
