from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .model_analyzer import (
    ModelAnalyzer,
    ATTENTION_LAYER_NAMES,
    FFN_LAYER_NAMES,
)
from .request import GenerationRequest


@dataclass
class AFDInstanceState:
    """Tracks the availability of one attention/FFN instance."""

    available_at: float = 0.0


class AFDAnalyzer:
    """
    Optional analyzer to estimate latency under Attention-FFN Disaggregated (AFD) decode.

    Prefill is unchanged. After prefill, all requests join a global waitlist that is
    pulled by attention instances. Each attention instance:
    - Pulls up to attention_batch_size requests when available.
    - Pays KV transfer once for the pulled requests.
    - Computes attention sequentially for the batch (per-layer).
    - Ships the activations as one block to a chosen FFN instance (A2F transfer).
    - The FFN instance performs continuous batching with max_batch_size; if more tokens
      are queued than the max, multiple FFN runs are issued back-to-back.
    - After FFN compute finishes for the batch, activations are sent back (F2A).
    - One layer duration for the batch is:
        max(attention_layer_time, ffn_layer_time_with_wait) + A2F + F2A
      and total decode latency is layer_duration * num_layers.
    Attention instances stay busy for the full multi-layer decode of the batch.
    """

    def __init__(
        self,
        model_analyzer: ModelAnalyzer,
        attention_batch_size: int = 6,
        ffn_max_batch_size: int = 6,
        num_attention_instances: int = 1,
        num_ffn_instances: int = 1,
        kv_transfer_bandwidth_gbps: float = 100.0,
        activation_bandwidth_gbps: float = 100.0,
        attention_analyzers: Optional[List[ModelAnalyzer]] = None,
        ffn_analyzers: Optional[List[ModelAnalyzer]] = None,
    ):
        self.model_analyzer = model_analyzer
        self.attention_batch_size = attention_batch_size
        self.ffn_max_batch_size = ffn_max_batch_size
        self.attention_analyzers = attention_analyzers or [
            model_analyzer for _ in range(max(1, num_attention_instances))
        ]
        self.ffn_analyzers = ffn_analyzers or [
            model_analyzer for _ in range(max(1, num_ffn_instances))
        ]
        self.num_attention_instances = max(1, len(self.attention_analyzers))
        self.num_ffn_instances = max(1, len(self.ffn_analyzers))

        self.kv_bandwidth_bytes = max(kv_transfer_bandwidth_gbps, 1e-6) * 1e9
        self.activation_bandwidth_bytes = max(activation_bandwidth_gbps, 1e-6) * 1e9

        self.hidden_size = self.model_analyzer.config.get_hidden_size(
            self.model_analyzer.model_params
        )
        self.num_layers = self.model_analyzer.config.get_num_hidden_layers(
            self.model_analyzer.model_params
        )

        # Availability timelines
        self.attn_states: List[AFDInstanceState] = [
            AFDInstanceState() for _ in range(self.num_attention_instances)
        ]
        self.ffn_states: List[AFDInstanceState] = [
            AFDInstanceState() for _ in range(self.num_ffn_instances)
        ]

        # Caches to avoid repeated analyzer calls
        self._layer_time_cache: Dict[Tuple[str, int, int, int], Tuple[float, float]] = {}

    @staticmethod
    def _seq_len_for_request(request: GenerationRequest) -> int:
        return max(request.input_length + request.generated_tokens, 1)

    def _get_layer_times(
        self,
        seq_len: int,
        batch_size: int,
        analyzer: Optional[ModelAnalyzer] = None,
        cache_role: str = "shared",
        instance_idx: int = 0,
    ) -> Tuple[float, float]:
        """
        Return (attention_time_per_layer_s, ffn_time_per_layer_s) for given seq_len/batch_size.
        """
        key = (cache_role, instance_idx, seq_len, batch_size)
        if key not in self._layer_time_cache:
            selected_analyzer = analyzer or self.model_analyzer
            breakdown = selected_analyzer.analyze_decode_breakdown(
                seqlen=seq_len, batchsize=batch_size
            )
            attn_time = breakdown["per_layer"]["attention_s"]
            ffn_time = breakdown["per_layer"]["ffn_s"]
            self._layer_time_cache[key] = (attn_time, ffn_time)
        return self._layer_time_cache[key]

    def _get_attention_layer_time(self, attn_idx: int, seq_len: int, batch_size: int) -> float:
        analyzer = self.attention_analyzers[attn_idx % len(self.attention_analyzers)]
        return self._get_layer_times(
            seq_len,
            batch_size,
            analyzer=analyzer,
            cache_role="attn",
            instance_idx=attn_idx,
        )[0]

    def _get_ffn_layer_time(self, ffn_idx: int, seq_len: int, batch_size: int) -> float:
        analyzer = self.ffn_analyzers[ffn_idx % len(self.ffn_analyzers)]
        return self._get_layer_times(
            seq_len,
            batch_size,
            analyzer=analyzer,
            cache_role="ffn",
            instance_idx=ffn_idx,
        )[1]

    def _kv_transfer_time(self, seq_len: int) -> float:
        """KV transfer time for one request."""
        kv_bytes = self._estimate_kv_cache_memory_for_length(seq_len)
        return kv_bytes / self.kv_bandwidth_bytes

    def _activation_transfer_time(self, batch_size: int) -> float:
        """Activation transfer time for A2F or F2A for a batch."""
        bytes_per_element = 2  # FP16
        act_bytes = self.hidden_size * batch_size * bytes_per_element
        return act_bytes / self.activation_bandwidth_bytes

    def _ffn_layer_time_for_batch(self, seq_len: int, batch_size: int, ffn_idx: int = 0) -> float:
        """
        FFN compute time for one layer for this batch, respecting FFN max_batch_size.
        If batch_size exceeds ffn_max_batch_size, multiple FFN runs are summed.
        """
        remaining = batch_size
        total_time = 0.0
        while remaining > 0:
            run_bsz = min(remaining, self.ffn_max_batch_size)
            total_time += self._get_ffn_layer_time(ffn_idx, seq_len, run_bsz)
            remaining -= run_bsz
        return total_time

    def _estimate_kv_cache_memory_for_length(self, sequence_length: int) -> float:
        """Match the engine's simplified KV estimate (bytes)."""
        hidden_size = getattr(self.model_analyzer.model_params, "hidden_size", 4096)
        num_layers = getattr(self.model_analyzer.model_params, "num_hidden_layers", 32)
        num_heads = getattr(self.model_analyzer.model_params, "num_attention_heads", 32)
        num_kv_heads = getattr(self.model_analyzer.model_params, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads
        bytes_per_element = 2  # FP16
        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
        return kv_per_token * max(sequence_length, 1)

    def reset(self):
        """Reset cached availability for a fresh simulation."""
        for state in self.attn_states:
            state.available_at = 0.0
        for state in self.ffn_states:
            state.available_at = 0.0

    def analyze_decode_latency(
        self,
        requests: List[GenerationRequest],
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Simulate decode latency for a list of prefilled requests under AFD.

        Returns a dict with per-batch timing breakdowns and per-request completion time.
        """
        waitlist: Deque[GenerationRequest] = deque(requests)
        batch_reports: List[Dict[str, Any]] = []
        completion_times: Dict[str, float] = {}

        while waitlist:
            # Choose the earliest-available attention instance
            attn_idx = min(
                range(len(self.attn_states)),
                key=lambda i: self.attn_states[i].available_at,
            )
            attn_start = max(start_time, self.attn_states[attn_idx].available_at)

            # Pull a batch from the global waitlist
            batch: List[GenerationRequest] = []
            while waitlist and len(batch) < self.attention_batch_size:
                batch.append(waitlist.popleft())
            if not batch:
                break

            seq_lens = [self._seq_len_for_request(req) for req in batch]
            max_seq_len = max(seq_lens)

            kv_time = sum(self._kv_transfer_time(seq) for seq in seq_lens)
            # Batched attention: use batch_size=len(batch) and the slowest (longest) request
            attn_layer_time = self._get_attention_layer_time(
                attn_idx, seq_len=max_seq_len, batch_size=len(batch)
            )

            # Choose the FFN instance that becomes free the earliest
            ffn_idx = min(
                range(len(self.ffn_states)),
                key=lambda i: self.ffn_states[i].available_at,
            )
            ffn_layer_time = self._ffn_layer_time_for_batch(max_seq_len, len(batch), ffn_idx)

            # Transfers per layer
            activation_transfer = self._activation_transfer_time(len(batch))
            earliest_ffn_start = (
                attn_start + kv_time + attn_layer_time + activation_transfer
            )
            ffn_wait = max(0.0, self.ffn_states[ffn_idx].available_at - earliest_ffn_start)
            effective_ffn_time = ffn_wait + ffn_layer_time

            per_layer_duration = (
                max(attn_layer_time, effective_ffn_time) + (2 * activation_transfer)
            )
            total_decode_time = kv_time + per_layer_duration * self.num_layers
            completion_time = attn_start + total_decode_time

            # Update timelines
            self.attn_states[attn_idx].available_at = completion_time
            self.ffn_states[ffn_idx].available_at = (
                max(self.ffn_states[ffn_idx].available_at, earliest_ffn_start)
                + ffn_layer_time
            )

            for req in batch:
                completion_times[req.req_id] = completion_time

            batch_reports.append(
                {
                    "batch_size": len(batch),
                    "attention_instance": attn_idx,
                    "ffn_instance": ffn_idx,
                    "request_ids": [req.req_id for req in batch],
                    "max_seq_len": max_seq_len,
                    "kv_transfer_time_s": kv_time,
                    "attention_layer_time_s": attn_layer_time,
                    "ffn_layer_time_s": ffn_layer_time,
                    "ffn_wait_s": ffn_wait,
                    "activation_transfer_s_per_direction": activation_transfer,
                    "per_layer_duration_s": per_layer_duration,
                    "num_layers": self.num_layers,
                    "total_decode_time_s": total_decode_time,
                    "start_time_s": attn_start,
                    "completion_time_s": completion_time,
                }
            )

        makespan = max(completion_times.values()) if completion_times else start_time

        return {
            "batches": batch_reports,
            "per_request_completion_s": completion_times,
            "makespan_s": makespan,
            "attention_batch_size": self.attention_batch_size,
            "ffn_max_batch_size": self.ffn_max_batch_size,
        }

    def next_attention_available_time(self) -> float:
        """Return earliest time any attention instance becomes free."""
        return min(state.available_at for state in self.attn_states)

    def schedule_next_batch(
        self,
        waitlist: Deque[GenerationRequest],
        start_time: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Consume up to attention_batch_size requests from the waitlist,
        schedule them on the earliest-available attention/FFN instances,
        and return a single-batch report plus per-request completion times.
        """
        if not waitlist:
            return None

        # Choose the earliest-available attention instance
        attn_idx = min(
            range(len(self.attn_states)),
            key=lambda i: self.attn_states[i].available_at,
        )
        attn_start = max(start_time, self.attn_states[attn_idx].available_at)

        # Pull a batch from the waitlist
        batch: List[GenerationRequest] = []
        while waitlist and len(batch) < self.attention_batch_size:
            batch.append(waitlist.popleft())
        if not batch:
            return None

        seq_lens = [self._seq_len_for_request(req) for req in batch]
        max_seq_len = max(seq_lens)
        max_output_len = max(getattr(req, "output_length", 1) for req in batch)

        kv_time = sum(self._kv_transfer_time(seq) for seq in seq_lens)
        # Batched attention: time driven by the longest sequence in the batch
        attn_layer_time = self._get_attention_layer_time(
            attn_idx, seq_len=max_seq_len, batch_size=len(batch)
        )

        # Choose the FFN instance that becomes free the earliest
        ffn_idx = min(
            range(len(self.ffn_states)),
            key=lambda i: self.ffn_states[i].available_at,
        )
        ffn_layer_time = self._ffn_layer_time_for_batch(max_seq_len, len(batch), ffn_idx)

        activation_transfer = self._activation_transfer_time(len(batch))
        earliest_ffn_start = (
            attn_start + kv_time + attn_layer_time + activation_transfer
        )
        ffn_wait = max(0.0, self.ffn_states[ffn_idx].available_at - earliest_ffn_start)
        effective_ffn_time = ffn_wait + ffn_layer_time

        per_layer_duration = (
            max(attn_layer_time, effective_ffn_time) + (2 * activation_transfer)
        )
        per_token_duration = per_layer_duration * self.num_layers
        total_decode_time = kv_time + per_token_duration * max(max_output_len, 1)
        completion_time = attn_start + total_decode_time

        # Update timelines
        self.attn_states[attn_idx].available_at = completion_time
        self.ffn_states[ffn_idx].available_at = (
            max(self.ffn_states[ffn_idx].available_at, earliest_ffn_start)
            + ffn_layer_time
        )

        completion_times = {req.req_id: completion_time for req in batch}
        batch_report = {
            "batch_size": len(batch),
            "attention_instance": attn_idx,
            "ffn_instance": ffn_idx,
            "request_ids": [req.req_id for req in batch],
            "max_seq_len": max_seq_len,
            "kv_transfer_time_s": kv_time,
            "attention_layer_time_s": attn_layer_time,
            "ffn_layer_time_s": ffn_layer_time,
            "ffn_wait_s": ffn_wait,
            "activation_transfer_s_per_direction": activation_transfer,
            "per_layer_duration_s": per_layer_duration,
            "per_token_duration_s": per_token_duration,
            "num_layers": self.num_layers,
            "total_decode_time_s": total_decode_time,
            "max_output_len": max_output_len,
            "start_time_s": attn_start,
            "completion_time_s": completion_time,
        }

        return {
            "batch": batch,
            "batch_report": batch_report,
            "completion_times": completion_times,
        }
