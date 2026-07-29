from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

from .request import GenerationRequest
from .engine import ServingEngine

@dataclass
class PlacementDecision:
    """Result of a placement decision."""
    target_engine: ServingEngine
    reason: str
    confidence: float = 1.0
    estimated_completion_time: Optional[float] = None

class PlacementAlgorithm(ABC):
    """Base class for request placement algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.statistics = {
            'total_decisions': 0,
            'rejections': 0,
            'placement_reasons': {}
        }

    @abstractmethod
    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        """
        Decide where to place a request.

        Args:
            request: The request to place
            available_engines: List of engines that can potentially handle the request

        Returns:
            PlacementDecision with target engine and reasoning, or None if no suitable engine found
        """
        pass

    def update_statistics(self, decision: Optional[PlacementDecision]):
        """Update algorithm statistics with placement decision."""
        self.statistics['total_decisions'] += 1
        if decision is None:
            self.statistics['rejections'] += 1
        else:
            reason = decision.reason
            self.statistics['placement_reasons'][reason] = \
                self.statistics['placement_reasons'].get(reason, 0) + 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm performance statistics."""
        return self.statistics.copy()


class RandomScheduler(PlacementAlgorithm):
    """Random placement algorithm - baseline for comparison."""

    def __init__(self):
        super().__init__("RandomScheduler")

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        target = random.choice(available_engines)
        decision = PlacementDecision(
            target_engine=target,
            reason="random_selection",
            confidence=0.1
        )
        self.update_statistics(decision)
        return decision


class RoundRobinScheduler(PlacementAlgorithm):
    """Round-robin placement algorithm - baseline for comparison."""

    def __init__(self):
        super().__init__("RoundRobinScheduler")
        self.current_index = 0

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        target = available_engines[self.current_index % len(available_engines)]
        self.current_index += 1

        decision = PlacementDecision(
            target_engine=target,
            reason="round_robin",
            confidence=0.2
        )
        self.update_statistics(decision)
        return decision


class OracleScheduler(PlacementAlgorithm):
    """
    Oracle scheduler with perfect knowledge of output length.
    Assigns requests to minimize estimated completion time.
    Upper bound on performance.
    """

    def __init__(self):
        super().__init__("OracleScheduler")

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        # Estimate completion time on each engine
        best_engine = None
        best_estimated_time = float('inf')
        best_load = float('inf')

        for engine in available_engines:
            # Rough estimate based on input/output tokens and hardware speed
            # Prefill time ~ proportional to input_length^2 / compute
            # Decode time ~ proportional to output_length / bandwidth
            prefill_work = request.input_length ** 1.5  # Simplified model
            decode_work = request.output_length * request.input_length  # KV cache access

            prefill_time = prefill_work / (engine.compute_throughput / 1e12)
            decode_time = decode_work / (engine.memory_bandwidth / 1e9)

            estimated_time = prefill_time + decode_time

            # Break ties by lowest load
            engine_load = engine.get_current_load()
            if estimated_time < best_estimated_time or (estimated_time == best_estimated_time and engine_load < best_load):
                best_estimated_time = estimated_time
                best_load = engine_load
                best_engine = engine

        decision = PlacementDecision(
            target_engine=best_engine,
            reason="oracle_minimum_completion_time",
            confidence=1.0,
            estimated_completion_time=best_estimated_time
        )
        self.update_statistics(decision)
        return decision


class FLOPsScheduler(PlacementAlgorithm):
    """
    Schedule purely by compute (FLOPs).
    Always routes to highest-compute GPU.
    Good for compute-bound workloads (long input).
    """

    def __init__(self):
        super().__init__("FLOPsScheduler")

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        # Select engine with highest compute throughput, break ties by lowest load
        best_engine = max(available_engines, key=lambda e: (e.compute_throughput, -e.get_current_load()))

        decision = PlacementDecision(
            target_engine=best_engine,
            reason="max_compute_throughput",
            confidence=0.6
        )
        self.update_statistics(decision)
        return decision


class BandwidthScheduler(PlacementAlgorithm):
    """
    Schedule purely by memory bandwidth.
    Always routes to highest-bandwidth GPU.
    Good for memory-bound workloads (long output).
    """

    def __init__(self):
        super().__init__("BandwidthScheduler")

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        # Select engine with highest memory bandwidth, break ties by lowest load
        best_engine = max(available_engines, key=lambda e: (e.memory_bandwidth, -e.get_current_load()))

        decision = PlacementDecision(
            target_engine=best_engine,
            reason="max_memory_bandwidth",
            confidence=0.6
        )
        self.update_statistics(decision)
        return decision


class RooflineScheduler(PlacementAlgorithm):
    """
    Use roofline model to determine if request is memory or compute bound.
    Treats entire request as one unit.
    """

    def __init__(self):
        super().__init__("RooflineScheduler")

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        # Calculate arithmetic intensity for entire request
        # Prefill: O(seq^2) ops, O(seq) memory
        # Decode: O(seq*output) ops, O(seq*output) memory
        prefill_ops = request.input_length ** 2 * 1e6  # Simplified
        decode_ops = request.input_length * request.output_length * 1e6
        total_ops = prefill_ops + decode_ops

        prefill_mem = request.input_length * 1e6
        decode_mem = request.input_length * request.output_length * 1e6
        total_mem = prefill_mem + decode_mem

        arithmetic_intensity = total_ops / total_mem if total_mem > 0 else 0

        # Select best engine based on roofline
        best_engine = None
        best_score = -1
        best_load = float('inf')

        for engine in available_engines:
            # Roofline turning point
            turning_point = engine.compute_throughput / engine.memory_bandwidth

            if arithmetic_intensity < turning_point:
                # Memory-bound → prioritize bandwidth
                score = engine.memory_bandwidth
                bound_type = "memory"
            else:
                # Compute-bound → prioritize compute
                score = engine.compute_throughput
                bound_type = "compute"

            # Break ties by lowest load
            engine_load = engine.get_current_load()
            if score > best_score or (score == best_score and engine_load < best_load):
                best_score = score
                best_load = engine_load
                best_engine = engine

        decision = PlacementDecision(
            target_engine=best_engine,
            reason=f"roofline_{bound_type}_bound",
            confidence=0.8
        )
        self.update_statistics(decision)
        return decision


class InputOutputAdaptiveScheduler_Roofline(PlacementAlgorithm):
    """
    Adaptive scheduler that analyzes prefill and decode separately.
    - Long input (prefill-heavy) → weight by compute
    - Long output (decode-heavy) → weight by bandwidth
    Uses roofline model for each phase.
    """

    def __init__(self):
        super().__init__("InputOutputAdaptiveScheduler_Roofline")

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        # Analyze prefill and decode separately
        prefill_ops = request.input_length ** 2 * 1e6
        prefill_mem = request.input_length * 1e6
        prefill_intensity = prefill_ops / prefill_mem if prefill_mem > 0 else 0

        decode_ops = request.input_length * request.output_length * 1e6
        decode_mem = request.input_length * request.output_length * 1e6
        decode_intensity = decode_ops / decode_mem if decode_mem > 0 else 0

        # Weight by input/output lengths
        total_work = request.input_length + request.output_length
        prefill_weight = request.input_length / total_work if total_work > 0 else 0.5
        decode_weight = request.output_length / total_work if total_work > 0 else 0.5

        # Score engines
        best_engine = None
        best_score = -1
        best_load = float('inf')

        for engine in available_engines:
            turning_point = engine.compute_throughput / engine.memory_bandwidth

            # Prefill score
            if prefill_intensity < turning_point:
                prefill_score = engine.memory_bandwidth
            else:
                prefill_score = engine.compute_throughput

            # Decode score
            if decode_intensity < turning_point:
                decode_score = engine.memory_bandwidth
            else:
                decode_score = engine.compute_throughput

            # Weighted combination
            score = prefill_weight * prefill_score + decode_weight * decode_score

            # Break ties by lowest load
            engine_load = engine.get_current_load()
            if score > best_score or (score == best_score and engine_load < best_load):
                best_score = score
                best_load = engine_load
                best_engine = engine

        decision = PlacementDecision(
            target_engine=best_engine,
            reason="input_output_adaptive_roofline",
            confidence=0.9
        )
        self.update_statistics(decision)
        return decision


class InputOutputAdaptiveScheduler_Threshold(PlacementAlgorithm):
    """
    Threshold-based adaptive scheduler.
    - input > 1024 tokens → increase compute weight
    - output > 512 tokens → increase bandwidth weight
    Blends both scores based on request characteristics.
    """

    def __init__(self, input_threshold=1024, output_threshold=512):
        super().__init__("InputOutputAdaptiveScheduler_Threshold")
        self.input_threshold = input_threshold
        self.output_threshold = output_threshold

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        # Calculate weights based on thresholds
        compute_weight = 1.0 if request.input_length > self.input_threshold else 0.3
        bandwidth_weight = 1.0 if request.output_length > self.output_threshold else 0.3

        # Score engines
        best_engine = None
        best_score = -1
        best_load = float('inf')

        for engine in available_engines:
            compute_score = engine.compute_throughput
            bandwidth_score = engine.memory_bandwidth

            # Weighted combination
            total_weight = compute_weight + bandwidth_weight
            score = (compute_weight * compute_score + bandwidth_weight * bandwidth_score) / total_weight

            # Break ties by lowest load
            engine_load = engine.get_current_load()
            if score > best_score or (score == best_score and engine_load < best_load):
                best_score = score
                best_load = engine_load
                best_engine = engine

        reason = []
        if request.input_length > self.input_threshold:
            reason.append("long_input")
        if request.output_length > self.output_threshold:
            reason.append("long_output")
        reason_str = "_".join(reason) if reason else "short_request"

        decision = PlacementDecision(
            target_engine=best_engine,
            reason=f"threshold_adaptive_{reason_str}",
            confidence=0.85
        )
        self.update_statistics(decision)
        return decision


class HeterogeneousRiskAwareScheduler(PlacementAlgorithm):
    """
    Risk-Aware Bin Packing scheduler for heterogeneous clusters.
    - Tier 1: H100 (FP8, High Compute)
    - Tier 2: A100 (FP16, Generalist)
    - Tier 3: RTX 3090 (FP16, Memory Constrained)
    
    Implements:
    1. Hard Constraint Filtering (OOM Prevention for Tier 3)
    2. Arithmetic Intensity Matching (Long Prefills -> Tier 1)
    3. Load Balancing (Least Memory Pressure)
    """

    def __init__(self, high_intensity_threshold: int = 2048, tier3_kv_safe_buffer: float = 0.5):
        super().__init__("HeterogeneousRiskAwareScheduler")
        self.high_intensity_threshold = high_intensity_threshold
        self.tier3_kv_safe_buffer = tier3_kv_safe_buffer

    def _get_tier(self, engine: ServingEngine) -> int:
        """Determine tier based on hardware."""
        hw = engine.hardware.lower()
        if "h100" in hw:
            return 1
        elif "a100" in hw:
            return 2
        elif "3090" in hw or "rtx" in hw:
            return 3
        return 2  # Default to Tier 2 for unknown

    def place_request(self, request: GenerationRequest,
                     available_engines: List[ServingEngine]) -> Optional[PlacementDecision]:
        if not available_engines:
            self.update_statistics(None)
            return None

        candidates = []
        rejection_reasons = {}

        # Step 1: Hard Constraint Filtering (OOM Prevention)
        # Estimate initial memory footprint: 2 * L_in * Layers * Hidden * 2 bytes (FP16)
        # We need model config for this. ServingEngine has model_params.
        
        for engine in available_engines:
            tier = self._get_tier(engine)
            
            # Check Tier 3 constraints
            if tier == 3:
                # Get model params from engine
                hidden_size = getattr(engine.model_params, 'hidden_size', 4096)
                num_layers = getattr(engine.model_params, 'num_hidden_layers', 32)
                
                # Estimate KV memory for input only (initial footprint)
                # KV cache per token: 2 * num_layers * hidden_size * 2 bytes (approx)
                # Note: This is a simplified view. Accurate calculation is in engine._estimate_kv_cache_memory
                # But we use the logic from the design doc: Mem_init > 50% of KV_safe
                
                # Let's use the engine's estimation method if possible, or approximate
                est_mem_init = engine._estimate_kv_cache_memory(request)
                
                # KV_safe is the *available* KV budget. 
                # Design doc says: "KV_safe be the safe KV-cache budget for 3090 (e.g., 6GB reserved for cache)"
                # "If Mem_init > 50% of KV_safe, Hard Block"
                
                # We'll use the engine's total memory capacity and subtract model weights to get total KV budget
                mem_info = engine.get_memory_info()
                total_kv_budget = mem_info['total'] - mem_info['model_weights']
                
                # If we assume some reserve, say 80% is usable for KV
                kv_safe = total_kv_budget * 0.8
                
                if est_mem_init > (kv_safe * self.tier3_kv_safe_buffer):
                    rejection_reasons[engine.engine_id] = "tier3_oom_risk"
                    continue

            candidates.append(engine)

        if not candidates:
            # If all rejected due to constraints, try to fallback to Tier 1/2 if they were filtered out?
            # The filter was only for Tier 3. If Tier 1/2 were available, they should be in candidates.
            # If candidates is empty, it means no engines available or all were Tier 3 and got blocked.
            self.update_statistics(None)
            return None

        # Step 2: Arithmetic Intensity Matching
        # If L_in > Threshold_High, prioritize Tier 1
        is_high_intensity = request.input_length > self.high_intensity_threshold
        
        priority_candidates = []
        if is_high_intensity:
            priority_candidates = [e for e in candidates if self._get_tier(e) == 1]
        
        # If no Tier 1 available (or not high intensity), fall back to all valid candidates
        final_candidates = priority_candidates if priority_candidates else candidates

        # Step 3: Load Balancing (Least Memory Pressure)
        # Score = (Used_KV + New_KV) / Total_KV
        best_engine = None
        best_score = float('inf')

        for engine in final_candidates:
            mem_info = engine.get_memory_info()
            # Total KV capacity (approximate as total - model weights)
            total_kv_capacity = mem_info['total'] - mem_info['model_weights']
            if total_kv_capacity <= 0:
                continue

            current_kv = mem_info['kv_cache']
            new_kv = engine._estimate_kv_cache_memory(request)
            
            score = (current_kv + new_kv) / total_kv_capacity
            
            # Break ties by current load (request count)
            if score < best_score:
                best_score = score
                best_engine = engine
            elif score == best_score:
                if engine.get_current_load() < best_engine.get_current_load():
                    best_engine = engine

        if best_engine:
            decision = PlacementDecision(
                target_engine=best_engine,
                reason=f"risk_aware_tier{self._get_tier(best_engine)}",
                confidence=0.9
            )
            self.update_statistics(decision)
            return decision
        
        self.update_statistics(None)
        return None


def get_scheduler(name: str, **kwargs) -> PlacementAlgorithm:
    """Factory function to create scheduler instances."""
    schedulers = {
        'random': RandomScheduler,
        'round_robin': RoundRobinScheduler,
        'oracle': OracleScheduler,
        'flops': FLOPsScheduler,
        'bandwidth': BandwidthScheduler,
        'roofline': RooflineScheduler,
        'inputoutput_roofline': InputOutputAdaptiveScheduler_Roofline,
        'inputoutput_threshold': InputOutputAdaptiveScheduler_Threshold,
        'heterogeneous_risk_aware': HeterogeneousRiskAwareScheduler,
    }

    scheduler_class = schedulers.get(name.lower())
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")

    return scheduler_class(**kwargs)