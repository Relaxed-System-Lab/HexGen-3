"""Shared data structures for the HexGen-3 style scheduling framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple


WORKER_TYPES: Tuple[str, str, str] = ("pre", "attn", "ffn")


@dataclass(frozen=True)
class WorkloadProfile:
    """Sliding-window workload summary W = <lambda, mu_in, mu_out>."""

    arrival_rate: float
    input_lengths: Tuple[int, ...]
    output_lengths: Tuple[int, ...]
    max_batch_size: int = 8

    def __post_init__(self):
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if not self.input_lengths or not self.output_lengths:
            raise ValueError("input_lengths and output_lengths must be non-empty")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

    @property
    def mean_input(self) -> int:
        return max(1, int(round(sum(self.input_lengths) / len(self.input_lengths))))

    @property
    def mean_output(self) -> int:
        return max(1, int(round(sum(self.output_lengths) / len(self.output_lengths))))

    @property
    def mean_decode_context(self) -> int:
        # Average context seen by decode is prompt plus roughly half of output.
        return max(1, self.mean_input + max(1, self.mean_output // 2))

    @classmethod
    def synthetic(
        cls,
        arrival_rate: float = 4.0,
        short_input: int = 512,
        long_input: int = 4096,
        short_output: int = 128,
        long_output: int = 512,
        long_ratio: float = 0.25,
        samples: int = 64,
        max_batch_size: int = 8,
    ) -> "WorkloadProfile":
        long_count = int(round(samples * max(0.0, min(1.0, long_ratio))))
        short_count = max(0, samples - long_count)
        inputs = (short_input,) * short_count + (long_input,) * long_count
        outputs = (short_output,) * short_count + (long_output,) * long_count
        return cls(
            arrival_rate=arrival_rate,
            input_lengths=tuple(inputs),
            output_lengths=tuple(outputs),
            max_batch_size=max_batch_size,
        )


@dataclass(frozen=True)
class ParallelismReplica:
    """One DP replica with its own TP and EP degree."""

    tp: int
    ep: int = 1

    def __post_init__(self):
        if self.tp <= 0:
            raise ValueError("replica tp must be positive")
        if self.ep <= 0:
            raise ValueError("replica ep must be positive")

    @property
    def gpus(self) -> int:
        # The paper's C1 accounts for allocated GPUs by summing TP degrees.
        return self.tp

    def as_tuple(self) -> Tuple[int, int]:
        return (self.tp, self.ep)


@dataclass(frozen=True)
class ParallelismStrategy:
    """Hybrid strategy p with possibly non-uniform DP replicas.

    The public ``ParallelismStrategy(dp, tp, ep)`` form remains supported for
    existing callers. When ``replicas`` is supplied, ``dp`` is the replica count
    and ``gpus`` is ``sum(replica.tp)`` as in the HexGen-3 local-search
    constraint C1.
    """

    dp: int
    tp: int
    ep: int = 1
    replicas: Tuple[ParallelismReplica, ...] = field(default_factory=tuple)

    def __post_init__(self):
        replicas = self.replicas
        if not replicas:
            if self.dp <= 0 or self.tp <= 0 or self.ep <= 0:
                raise ValueError("dp, tp, and ep must be positive")
            replicas = tuple(ParallelismReplica(tp=self.tp, ep=self.ep) for _ in range(self.dp))
        else:
            normalized = []
            for replica in replicas:
                if isinstance(replica, ParallelismReplica):
                    normalized.append(replica)
                else:
                    tp, ep = replica
                    normalized.append(ParallelismReplica(tp=int(tp), ep=int(ep)))
            replicas = tuple(normalized)

        if not replicas:
            raise ValueError("strategy must have at least one replica")

        uniform_tp = all(replica.tp == replicas[0].tp for replica in replicas)
        uniform_ep = all(replica.ep == replicas[0].ep for replica in replicas)
        object.__setattr__(self, "replicas", replicas)
        object.__setattr__(self, "dp", len(replicas))
        object.__setattr__(self, "tp", replicas[0].tp if uniform_tp else max(r.tp for r in replicas))
        object.__setattr__(self, "ep", replicas[0].ep if uniform_ep else max(r.ep for r in replicas))

    @classmethod
    def from_replicas(cls, replicas: Iterable[ParallelismReplica]) -> "ParallelismStrategy":
        replica_tuple = tuple(replicas)
        if not replica_tuple:
            raise ValueError("replicas must be non-empty")
        return cls(
            dp=len(replica_tuple),
            tp=replica_tuple[0].tp,
            ep=replica_tuple[0].ep,
            replicas=replica_tuple,
        )

    @property
    def gpus(self) -> int:
        return sum(replica.gpus for replica in self.replicas)

    @property
    def is_uniform(self) -> bool:
        return (
            bool(self.replicas)
            and all(replica.tp == self.replicas[0].tp for replica in self.replicas)
            and all(replica.ep == self.replicas[0].ep for replica in self.replicas)
        )

    def as_tuple(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(replica.as_tuple() for replica in self.replicas)

    def as_dict(self) -> Dict[str, object]:
        return {
            "dp": self.dp,
            "tp": self.tp,
            "ep": self.ep,
            "gpus": self.gpus,
            "uniform": self.is_uniform,
            "replicas": [
                {"tp": replica.tp, "ep": replica.ep, "gpus": replica.gpus}
                for replica in self.replicas
            ],
        }


@dataclass(frozen=True)
class WorkerSlice:
    worker_type: str
    hardware: str

    def __post_init__(self):
        if self.worker_type not in WORKER_TYPES:
            raise ValueError(f"unknown worker_type {self.worker_type!r}")

    def as_tuple(self) -> Tuple[str, str]:
        return (self.worker_type, self.hardware)


@dataclass
class AllocationMatrix:
    """Allocation matrix A[worker_type][hardware] -> GPU count."""

    values: Dict[str, Dict[str, int]]

    @classmethod
    def zeros(cls, hardware_types: Iterable[str]) -> "AllocationMatrix":
        return cls({t: {h: 0 for h in hardware_types} for t in WORKER_TYPES})

    @classmethod
    def uniform(cls, capacity: Mapping[str, int]) -> "AllocationMatrix":
        alloc = cls.zeros(capacity.keys())
        for hw, total in capacity.items():
            base, rem = divmod(max(0, int(total)), len(WORKER_TYPES))
            for worker in WORKER_TYPES:
                alloc.values[worker][hw] = base
            for worker in WORKER_TYPES[:rem]:
                alloc.values[worker][hw] += 1
        return alloc

    def clone(self) -> "AllocationMatrix":
        return AllocationMatrix({t: dict(hw) for t, hw in self.values.items()})

    def hardware_types(self) -> Tuple[str, ...]:
        seen = []
        for worker in WORKER_TYPES:
            for hw in self.values.get(worker, {}):
                if hw not in seen:
                    seen.append(hw)
        return tuple(seen)

    def get(self, worker_type: str, hardware: str) -> int:
        return int(self.values.get(worker_type, {}).get(hardware, 0))

    def set(self, worker_type: str, hardware: str, value: int) -> None:
        self.values.setdefault(worker_type, {})[hardware] = max(0, int(value))

    def total_for_hardware(self, hardware: str) -> int:
        return sum(self.get(worker, hardware) for worker in WORKER_TYPES)

    def total_gpus(self) -> int:
        return sum(sum(hw.values()) for hw in self.values.values())

    def unused_for_hardware(self, capacity: Mapping[str, int], hardware: str) -> int:
        return max(0, int(capacity.get(hardware, 0)) - self.total_for_hardware(hardware))

    def total_unused(self, capacity: Mapping[str, int]) -> int:
        return sum(self.unused_for_hardware(capacity, hardware) for hardware in capacity)

    def add(self, dst: str, hardware: str, block_size: int, capacity: Mapping[str, int]) -> bool:
        if block_size <= 0 or dst not in WORKER_TYPES:
            return False
        if self.unused_for_hardware(capacity, hardware) < block_size:
            return False
        self.set(dst, hardware, self.get(dst, hardware) + block_size)
        return True

    def move(self, src: str, dst: str, hardware: str, block_size: int) -> bool:
        if src == dst or block_size <= 0:
            return False
        if self.get(src, hardware) < block_size:
            return False
        self.set(src, hardware, self.get(src, hardware) - block_size)
        self.set(dst, hardware, self.get(dst, hardware) + block_size)
        return True

    def validate(self, capacity: Mapping[str, int]) -> None:
        for hw, total in capacity.items():
            used = self.total_for_hardware(hw)
            if used > total:
                raise ValueError(f"allocation overuses {hw}: {used} > {total}")
        for worker in WORKER_TYPES:
            for hw, count in self.values.get(worker, {}).items():
                if count < 0:
                    raise ValueError(f"negative allocation for {worker}/{hw}")

    def as_key(self) -> Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]:
        return tuple(
            (worker, tuple(sorted(self.values.get(worker, {}).items())))
            for worker in WORKER_TYPES
        )


@dataclass
class SlicePlan:
    worker_type: str
    hardware: str
    gpus: int
    strategy: ParallelismStrategy | None
    throughput: float
    reconfiguration_cost_s: float = 0.0
    score: float = 0.0


@dataclass
class ThroughputProfile:
    by_worker: Dict[str, float]

    @property
    def bottleneck(self) -> float:
        return min(self.by_worker.get(worker, 0.0) for worker in WORKER_TYPES)

    def as_key(self) -> Tuple[Tuple[str, float], ...]:
        return tuple((w, round(self.by_worker.get(w, 0.0), 12)) for w in WORKER_TYPES)


@dataclass
class DeploymentPlan:
    allocation: AllocationMatrix
    parallelism: Dict[Tuple[str, str], ParallelismStrategy]
    slice_plans: Dict[Tuple[str, str], SlicePlan]
    throughput: ThroughputProfile
    estimated_latency_s: float
    cost_per_hour: float
    req_per_dollar: float
    tail_latency_s: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)
