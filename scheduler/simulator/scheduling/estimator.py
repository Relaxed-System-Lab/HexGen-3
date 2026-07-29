"""Cost, throughput, and latency estimation for scheduling candidates."""

from __future__ import annotations

from functools import lru_cache
from math import inf
from random import Random
from statistics import mean
from typing import Dict, List, Mapping, Tuple

from simulator.configs.hardware import hardware_params
from simulator.configs.models import llama
from simulator.core.model_analyzer import ModelAnalyzer

from .types import (
    AllocationMatrix,
    DeploymentPlan,
    ParallelismStrategy,
    ThroughputProfile,
    WORKER_TYPES,
    WorkloadProfile,
)


class SimulatorEstimator:
    """Thin estimator wrapper around the copied simulator's ModelAnalyzer.

    The scheduler calls this many times, so estimates are cached by hardware,
    tensor-parallel degree, worker type, and workload summary.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        kv_transfer_bandwidth_gbps: float = 100.0,
        activation_bandwidth_gbps: float = 100.0,
    ):
        self.model_id = model_id
        self.kv_bandwidth_bytes = max(kv_transfer_bandwidth_gbps, 1e-6) * 1e9
        self.activation_bandwidth_bytes = max(activation_bandwidth_gbps, 1e-6) * 1e9
        self._analyzers: Dict[Tuple[str, int], ModelAnalyzer] = {}

    def _get_analyzer(self, hardware: str, tp_size: int) -> ModelAnalyzer:
        key = (hardware, tp_size)
        if key not in self._analyzers:
            self._analyzers[key] = ModelAnalyzer(
                model_id=self.model_id,
                config=llama,
                hardware=hardware,
            )
        return self._analyzers[key]

    def _model_params(self):
        # Any analyzer has the same model params. H100 is just a stable default.
        hw = "NVDA:H100:SXM" if "NVDA:H100:SXM" in hardware_params else next(iter(hardware_params))
        return self._get_analyzer(hw, 1).model_params

    def _num_layers(self) -> int:
        return int(getattr(self._model_params(), "num_hidden_layers", 32))

    def _hidden_size(self) -> int:
        return int(getattr(self._model_params(), "hidden_size", 4096))

    def _num_heads(self) -> int:
        return int(getattr(self._model_params(), "num_attention_heads", 32))

    def _num_kv_heads(self) -> int:
        return int(getattr(self._model_params(), "num_key_value_heads", self._num_heads()))

    def _kv_transfer_time(self, sequence_length: int, batch_size: int) -> float:
        hidden_size = self._hidden_size()
        num_layers = self._num_layers()
        num_heads = self._num_kv_heads()
        head_dim = max(1, hidden_size // max(1, self._num_heads()))
        bytes_per_element = 2
        kv_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_element
        return (kv_per_token * max(sequence_length, 1) * max(batch_size, 1)) / self.kv_bandwidth_bytes

    def _activation_transfer_time(self, batch_size: int, sequence_length: int = 1) -> float:
        bytes_per_element = 2
        act_bytes = self._hidden_size() * max(batch_size, 1) * max(sequence_length, 1) * bytes_per_element
        return act_bytes / self.activation_bandwidth_bytes

    def _effective_batch_size(self, tp: int) -> int:
        return max(1, 4 * max(tp, 1))

    @lru_cache(maxsize=8192)
    def _replica_throughput_cached(
        self,
        worker_type: str,
        hardware: str,
        tp: int,
        ep: int,
        mean_input: int,
        mean_output: int,
        mean_decode_context: int,
        max_batch_size: int,
    ) -> float:
        workload = WorkloadProfile(
            arrival_rate=1.0,
            input_lengths=(mean_input,),
            output_lengths=(mean_output,),
            max_batch_size=max_batch_size,
        )
        batch = self._effective_batch_size(tp)
        analyzer = self._get_analyzer(hardware, max(1, tp))

        try:
            if worker_type == "pre":
                result = analyzer.analyze(
                    seqlen=mean_input,
                    batchsize=batch,
                    w_bit=16,
                    a_bit=16,
                    kv_bit=16,
                    tp_size=max(1, tp),
                )
                batch_time = float(result["total_results"]["prefill"]["inference_time"])
            else:
                breakdown = analyzer.analyze_decode_breakdown(
                    seqlen=mean_decode_context,
                    batchsize=batch,
                    w_bit=16,
                    a_bit=16,
                    kv_bit=16,
                    tp_size=max(1, tp),
                )
                per_layer = breakdown["per_layer"]
                num_layers = int(breakdown.get("num_layers", self._num_layers()))
                a2f_time = self._activation_transfer_time(batch)
                f2a_time = self._activation_transfer_time(batch)
                if worker_type == "attn":
                    per_token_time = (
                        float(per_layer["attention_s"]) + a2f_time
                    ) * num_layers
                    batch_time = self._kv_transfer_time(mean_input, batch) + mean_output * per_token_time
                elif worker_type == "ffn":
                    ffn_time = float(per_layer["ffn_s"])
                    if ep > 1:
                        # EP only matters for MoE-style FFN shards. The copied
                        # analyzer has dense FFN kernels, so this is a bounded
                        # approximation with a small synchronization overhead.
                        ffn_time = ffn_time / ep + self._activation_transfer_time(batch) * 0.05 * (ep - 1)
                    per_token_time = (
                        ffn_time + f2a_time
                    ) * num_layers
                    batch_time = mean_output * per_token_time
                else:
                    raise ValueError(f"unknown worker_type {worker_type!r}")
        except Exception:
            return 0.0

        if batch_time <= 0:
            return 0.0
        return max(0.0, batch / batch_time)

    def estimate_slice_throughput(
        self,
        worker_type: str,
        hardware: str,
        strategy: ParallelismStrategy,
        workload: WorkloadProfile,
    ) -> float:
        if strategy.dp <= 0 or strategy.gpus <= 0:
            return 0.0
        return sum(
            self._replica_throughput_cached(
                worker_type,
                hardware,
                replica.tp,
                replica.ep,
                workload.mean_input,
                workload.mean_output,
                workload.mean_decode_context,
                workload.max_batch_size,
            )
            for replica in strategy.replicas
        )

    def estimate_latency(self, workload: WorkloadProfile, throughput: ThroughputProfile) -> float:
        tails = self.estimate_tail_latency(workload, throughput)
        if tails:
            return tails.get("mean", inf)
        bottleneck = throughput.bottleneck
        if bottleneck <= 0:
            return inf
        stage_service = sum(
            1.0 / max(throughput.by_worker.get(worker, 0.0), 1e-9)
            for worker in WORKER_TYPES
        )
        utilization = workload.arrival_rate / bottleneck
        if utilization >= 1.0:
            overload_delay = (utilization - 1.0) * max(workload.mean_output, 1)
            return stage_service + overload_delay + (1.0 / bottleneck)
        queue_delay = utilization / max(1e-9, (1.0 - utilization) * bottleneck)
        return stage_service + queue_delay

    def estimate_tail_latency(
        self,
        workload: WorkloadProfile,
        throughput: ThroughputProfile,
        samples: int = 512,
        seed: int = 13,
    ) -> Dict[str, float]:
        if throughput.bottleneck <= 0:
            return {"mean": inf, "p50": inf, "p95": inf, "p99": inf}

        request_shapes = self._sample_request_shapes(workload, samples)
        rng = Random(seed)
        stage_available = {worker: 0.0 for worker in WORKER_TYPES}
        arrival_time = 0.0
        latencies: List[float] = []

        for input_len, output_len in request_shapes:
            arrival_time += rng.expovariate(workload.arrival_rate)
            completion_time = arrival_time
            for worker in WORKER_TYPES:
                service = self._stage_service_time(worker, input_len, output_len, workload, throughput)
                start_time = max(completion_time, stage_available[worker])
                finish_time = start_time + service
                stage_available[worker] = finish_time
                completion_time = finish_time
            latencies.append(completion_time - arrival_time)

        latencies.sort()
        return {
            "mean": mean(latencies),
            "p50": self._percentile(latencies, 0.50),
            "p95": self._percentile(latencies, 0.95),
            "p99": self._percentile(latencies, 0.99),
        }

    def _sample_request_shapes(self, workload: WorkloadProfile, samples: int) -> List[Tuple[int, int]]:
        pairs = list(zip(workload.input_lengths, workload.output_lengths))
        if not pairs:
            return [(workload.mean_input, workload.mean_output)]
        return [pairs[i % len(pairs)] for i in range(max(1, samples))]

    def _stage_service_time(
        self,
        worker: str,
        input_len: int,
        output_len: int,
        workload: WorkloadProfile,
        throughput: ThroughputProfile,
    ) -> float:
        theta = max(throughput.by_worker.get(worker, 0.0), 1e-9)
        if worker == "pre":
            scale = max(input_len, 1) / max(workload.mean_input, 1)
        elif worker == "attn":
            context = input_len + max(1, output_len // 2)
            scale = context / max(workload.mean_decode_context, 1)
        else:
            scale = max(output_len, 1) / max(workload.mean_output, 1)
        return max(0.0, scale / theta)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return inf
        index = min(len(values) - 1, max(0, int(round(percentile * (len(values) - 1)))))
        return values[index]

    def estimate_cost_per_hour(self, allocation: AllocationMatrix) -> float:
        total = 0.0
        for hw in allocation.hardware_types():
            price = float(hardware_params.get(hw, {}).get("price_per_hour", 0.0))
            total += price * allocation.total_for_hardware(hw)
        return total

    def summarize_plan_cost(self, plan: DeploymentPlan) -> None:
        cost = self.estimate_cost_per_hour(plan.allocation)
        plan.cost_per_hour = cost
        plan.req_per_dollar = plan.throughput.bottleneck / (cost / 3600.0) if cost > 0 else 0.0
