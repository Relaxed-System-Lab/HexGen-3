from collections import deque
from transformers import AutoConfig
from typing import Any, Deque, Dict, List, Optional, Set
import numpy as np

from simulator.configs.hardware import hardware_params
from .model_analyzer import ModelAnalyzer
from .memory import MemoryPlanner
from .config import ParallelConfig
from .request import GenerationRequest, REQ_STATUS
from .trace import TraceEvent
from .events import Event, EventType, EventPriority


class Batch:
    """Represents a batch of requests being processed together."""

    def __init__(self, batch_id: str, requests: List[GenerationRequest]):
        self.batch_id = batch_id
        self.requests = requests
        self.created_at = 0.0
        self.status = "forming"  # forming, prefill, decode, completed
        self.memory_usage = 0.0

    def add_request(self, request: GenerationRequest):
        """Add a request to the batch."""
        if request not in self.requests:
            self.requests.append(request)

    def remove_request(self, request: GenerationRequest):
        """Remove a completed request from the batch."""
        if request in self.requests:
            self.requests.remove(request)

        # Mark batch as completed if empty
        if len(self.requests) == 0:
            self.status = "completed"

    def is_empty(self) -> bool:
        """Check if batch is empty (all requests completed)."""
        return len(self.requests) == 0

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return len(self.requests)


class ServingEngine:
    """Event-driven serving engine for local node simulation."""

    def __init__(
            self,
            engine_id: str,
            model_id: str,
            model_instance: Any,
            hardware: str,
            parallel_config=None,
            max_batch_size: Optional[int] = None,
            disable_attention: bool = False,
            disable_ffn: bool = False,
            pd_separation: bool = False,
            pd_prefill_only: bool = False,
            pd_decode_only: bool = False,
            kv_transfer_bandwidth_gbps: float = 100.0,
            afd_enabled: bool = False,
            afd_attention: bool = False,
            afd_ffn: bool = False,
        ):
        """
        Initialize the serving engine with model and hardware configurations.

        Args:
            engine_id: Unique identifier for this engine
            model_id: Identifier for the model to be served
            model_instance: Configuration object for the model
            hardware: Hardware type identifier
            parallel_config: Optional parallel configuration for the engine
            max_batch_size: Maximum batch size for processing
            disable_attention: If True, disable attention computation
            disable_ffn: If True, disable FFN computation
        """
        self.engine_id = engine_id
        self.model_id = model_id
        
        # Load model config with rope_scaling compatibility fix for Llama 3.1
        # Patch the validation to allow llama3 rope_scaling type when available.
        from transformers.models.llama.configuration_llama import LlamaConfig
        original_rope_validation = getattr(LlamaConfig, "_rope_scaling_validation", None)

        if original_rope_validation is not None:
            def patched_rope_validation(self):
                """Patched version that allows llama3 rope_type for Llama 3.1."""
                if self.rope_scaling is None:
                    return
                if not isinstance(self.rope_scaling, dict):
                    raise ValueError(f"`rope_scaling` should be a dictionary, got {type(self.rope_scaling)}")

                if self.rope_scaling.get("rope_type") == "llama3":
                    # Allow llama3 format without raising.
                    return

                return original_rope_validation(self)

            # Monkey-patch the validation
            LlamaConfig._rope_scaling_validation = patched_rope_validation
        
        try:
            self.model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load model config: {e}")
            # Fallback: create minimal config
            self.model_params = LlamaConfig(
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                vocab_size=128256,
                rope_scaling={
                    "type": "dynamic",
                    "factor": 8.0
                }
            )
        
        self.model_instance = model_instance
        self.hardware = hardware
        self.parallel_config = parallel_config
        self.max_batch_size = max_batch_size if max_batch_size is not None else 1_000_000
        self.disable_attention = disable_attention
        self.disable_ffn = disable_ffn
        self.pd_separation = pd_separation
        self.pd_prefill_only = pd_prefill_only
        self.pd_decode_only = pd_decode_only
        self.kv_transfer_bandwidth_gbps = kv_transfer_bandwidth_gbps
        self.afd_enabled = afd_enabled
        self.afd_attention = afd_attention
        self.afd_ffn = afd_ffn

        # Hardware specifications
        self.hardware_spec = hardware_params[hardware]
        self.gpu_memory_capacity = self.hardware_spec['vmemory']
        self.memory_bandwidth = self.hardware_spec['bandwidth']
        self.compute_throughput = self.hardware_spec['FP16']

        # Memory planning (preferred) with heuristic fallback if unavailable
        self.memory_planner: Optional[MemoryPlanner] = None

        # Request management
        self.request_queue: Deque[GenerationRequest] = deque()
        self.active_requests: Set[GenerationRequest] = set()
        self.completed_requests: List[GenerationRequest] = []

        # Separate queues for different processing stages
        self.prefill_queue: Deque[GenerationRequest] = deque()  # Requests waiting for prefill
        self.decode_ready_requests: Deque[GenerationRequest] = deque()  # Requests ready for decode batching

        # Batch management (for decode phase only)
        self.current_decode_batch: Optional[Batch] = None
        self.batch_counter = 0

        # Current prefill request (processed individually)
        self.current_prefill_request: Optional[GenerationRequest] = None

        # Tracing
        self.trace_events: List[TraceEvent] = []
        self.current_time = 0.0  # Last wall-clock time passed in from cluster
        self.time_cursor = 0.0   # Engine's internal notion of GPU time progression

        # Model analyzer for accurate timing
        # Use appropriate config based on model_id
        analyzer_config = model_instance
        if analyzer_config is None:
            # Try to provide appropriate config based on model_id
            if "llama" in self.model_id.lower() or "Llama-2" in self.model_id:
                try:
                    from simulator.configs.models import llama
                    analyzer_config = llama
                except ImportError:
                    analyzer_config = None
            else:
                analyzer_config = None

        self.model_analyzer = ModelAnalyzer(
            model_id=self.model_id,
            config=analyzer_config,
            hardware=self.hardware
        )

        if analyzer_config is not None:
            try:
                self.memory_planner = MemoryPlanner(
                    model_params=self.model_params,
                    model_config=analyzer_config,
                    w_bit=16,
                    a_bit=16,
                    kv_bit=16,
                    hardware_params=self.hardware_spec,
                    parallel_config=parallel_config,
                    block_size=16,
                )
            except Exception as e:
                print(f"Warning: failed to initialize MemoryPlanner on {self.engine_id}: {e}")
                self.memory_planner = None

        # Model loading state
        self.model_loaded = False
        self.model_memory_usage = 0.0
        self.kv_cache_memory = 0.0
        self.kv_cache_capacity = 0.0

        # Performance tracking
        self.statistics = {
            'requests_processed': 0,
            'prefill_requests_processed': 0,
            'total_tokens_generated': 0,
            'total_prefill_time': 0.0,
            'total_decode_time': 0.0,
            'memory_peak': 0.0,
            'batch_sizes': []
        }

        # Event callback (set by cluster manager)
        self.event_callback = None

    def set_event_callback(self, callback):
        """Set callback function to emit events to cluster manager."""
        self.event_callback = callback

    def emit_event(self, event_type: EventType, data: Dict[str, Any],
                   priority: EventPriority = EventPriority.MEDIUM):
        """Emit an event to the cluster manager."""
        if self.event_callback:
            event = Event(
                timestamp=0.0,  # Will be set by cluster manager
                event_type=event_type,
                target=self.engine_id,
                data=data,
                priority=priority
            )
            self.event_callback(event)

    def _refresh_memory_usage(self):
        """Refresh cached memory usage from planner (if enabled) or heuristics."""
        if self.memory_planner:
            try:
                self.kv_cache_memory = self.memory_planner.get_allocated_kv_memory_per_shard()
                self.kv_cache_capacity = self.memory_planner.get_total_kv_memory_capacity_per_shard()
            except Exception:
                self.kv_cache_memory = 0.0
                self.kv_cache_capacity = 0.0
        else:
            self.kv_cache_capacity = 0.0

        total_used = self.model_memory_usage + self.kv_cache_memory
        self.statistics['memory_peak'] = max(self.statistics['memory_peak'], total_used)

    def load_model(self) -> bool:
        """Load the model into GPU memory."""
        if self.model_loaded:
            return True

        model_memory = 0.0
        if self.memory_planner:
            try:
                model_memory = self.memory_planner.get_weights_memory_per_shard()
            except Exception as e:
                print(f"Warning: MemoryPlanner weight estimate failed on {self.engine_id}: {e}")
                model_memory = 0.0

        if model_memory <= 0:
            # Heuristic fallback if planner is unavailable
            hidden_size = getattr(self.model_params, 'hidden_size', 4096)
            num_layers = getattr(self.model_params, 'num_hidden_layers', 32)
            vocab_size = getattr(self.model_params, 'vocab_size', 50257)

            total_params = (
                hidden_size * hidden_size * 4 * num_layers +  # MLP weights
                hidden_size * hidden_size * 3 * num_layers +  # Attention weights
                hidden_size * vocab_size * 2                  # Embedding and LM head
            )
            model_memory = total_params * 2  # FP16 bytes

        if model_memory > self.gpu_memory_capacity:
            return False

        self.model_memory_usage = model_memory
        self.model_loaded = True
        self._refresh_memory_usage()

        self.emit_event(
            EventType.MODEL_LOAD,
            {'model_id': self.model_id, 'memory_usage': model_memory},
            EventPriority.HIGH
        )

        return True

    def unload_model(self):
        """Unload the model from GPU memory."""
        if self.model_loaded:
            self.model_loaded = False
            self.model_memory_usage = 0.0
            self.kv_cache_memory = 0.0
            self.kv_cache_capacity = 0.0
            if self.memory_planner:
                self.memory_planner.reset_allocations()

            self.emit_event(
                EventType.MODEL_UNLOAD,
                {'model_id': self.model_id},
                EventPriority.HIGH
            )

    def can_accommodate_request(self, request: GenerationRequest,
                               safety_margin: float = 0.1) -> bool:
        """Check if the engine can accommodate a new request."""
        if not self.model_loaded:
            if not self.load_model():
                return False

        if self.memory_planner:
            if not self.memory_planner.can_allocate_request(request):
                return False
            current_kv = self.memory_planner.get_allocated_kv_memory_per_shard()
            additional_kv = self.memory_planner.estimate_additional_kv_memory_per_shard(request)
            total_memory_needed = self.model_memory_usage + current_kv + additional_kv
            available_memory = self.gpu_memory_capacity * (1 - safety_margin)
            return total_memory_needed <= available_memory

        # Heuristic fallback when planner is unavailable
        if self.pd_prefill_only:
            seq_len = max(request.input_length, 1)
            estimated_kv_memory = self._estimate_kv_cache_memory_for_length(seq_len)
            total_memory_needed = self.model_memory_usage + self.kv_cache_memory + estimated_kv_memory
        else:
            seq_len = max(request.input_length + max(request.generated_tokens, 1), 1)
            estimated_kv_memory = self._estimate_kv_cache_memory_for_length(seq_len)
            if self.pd_decode_only:
                total_memory_needed = self.model_memory_usage + estimated_kv_memory
            else:
                total_memory_needed = self.model_memory_usage + self.kv_cache_memory + estimated_kv_memory
        available_memory = self.gpu_memory_capacity * (1 - safety_margin)
        return total_memory_needed <= available_memory

    def set_pd_role(self, prefill_only: Optional[bool] = None, decode_only: Optional[bool] = None):
        """Dynamically change PD role for this engine."""
        if prefill_only is not None:
            self.pd_prefill_only = prefill_only
        if decode_only is not None:
            self.pd_decode_only = decode_only

    def add_request(self, request: GenerationRequest):
        """Add a new request to the processing queue."""
        if self.pd_decode_only:
            # Decode-only engines should not accept fresh arrivals
            return False
        if not self.can_accommodate_request(request):
            return False

        if self.memory_planner and not self.memory_planner.has_allocation(request.req_id):
            self.memory_planner.allocate(request)
            self._refresh_memory_usage()

        # New requests go directly to prefill queue since prefill has priority
        self.prefill_queue.append(request)
        request.status = REQ_STATUS.SCHEDULED

        # Don't emit REQUEST_ARRIVAL event here since the cluster manager
        # already knows about this request and is the one adding it

        return True

    def add_decode_ready_request(self, request: GenerationRequest, ready_time: float = 0.0):
        """Add a request that has already completed prefill elsewhere."""
        if self.pd_prefill_only:
            return False
        if not self.can_accommodate_request(request):
            return False

        if self.memory_planner and not self.memory_planner.has_allocation(request.req_id):
            self.memory_planner.allocate(request)
            self._refresh_memory_usage()

        # Align engine time with incoming ready time
        self.set_current_time(max(self.current_time, ready_time))

        request.status = REQ_STATUS.GENERATE
        self.active_requests.add(request)
        self.decode_ready_requests.append(request)
        return True

    def get_current_load(self) -> int:
        """Get current number of active requests."""
        return len(self.active_requests) + len(self.decode_ready_requests) + (1 if self.current_prefill_request else 0)

    def has_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is loaded."""
        return self.model_loaded and self.model_id == model_id

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        self._refresh_memory_usage()
        total_used = self.model_memory_usage + self.kv_cache_memory
        info = {
            'total': self.gpu_memory_capacity,
            'model_weights': self.model_memory_usage,
            'kv_cache': self.kv_cache_memory,
            'kv_cache_capacity': self.kv_cache_capacity,
            'available': self.gpu_memory_capacity - total_used,
            'used': total_used,
            'utilization': total_used / self.gpu_memory_capacity
        }

        if self.memory_planner:
            info['kv_blocks_used'] = self.memory_planner.get_allocated_block_count()
            info['kv_blocks_capacity'] = self.memory_planner.get_max_block_count()

        return info

    def step(self, current_time: Optional[float] = None) -> List[Event]:
        """Execute one step of the serving engine with priority-based processing."""
        # Update current time if provided
        if current_time is not None:
            self.current_time = current_time
            # Ensure GPU timeline never goes backwards
            if self.time_cursor < self.current_time:
                self.time_cursor = self.current_time

        events = []

        # In AFD mode decode-only nodes are orchestrated by the cluster-level scheduler
        if self.afd_enabled and self.pd_decode_only:
            return events

        # Priority 1: Process prefill requests (individual processing, no batching)
        if self.current_prefill_request is None and self.prefill_queue:
            # Start prefill for next waiting request
            self._start_next_prefill()

        if self.current_prefill_request:
            # Continue processing current prefill request
            prefill_complete = self._process_individual_prefill(self.current_time)
            if prefill_complete:
                # Move completed prefill request to decode-ready queue (only when not PD-split)
                if not self.pd_separation:
                    self.decode_ready_requests.append(self.current_prefill_request)
                self.current_prefill_request = None

        # Priority 2: Handle decode batch formation and processing
        # Form decode batch whenever we have requests ready for decode and no current decode batch
        # Continuous batching: requests can join decode as soon as they complete prefill
        if self.current_decode_batch is None and self.decode_ready_requests:
            self._form_decode_batch()
        elif self.current_decode_batch and self.decode_ready_requests:
            # Add new ready requests to existing decode batch (continuous batching)
            self._add_to_decode_batch()

        if self.current_decode_batch and not self.current_decode_batch.is_empty():
            decode_events = self._process_decode_batch(self.current_time)
            events.extend(decode_events)

        # Clean up empty decode batch
        if self.current_decode_batch and self.current_decode_batch.is_empty():
            self.current_decode_batch = None

        return events

    def _start_next_prefill(self):
        """Start prefill for the next request in the prefill queue."""
        if not self.prefill_queue:
            return

        # Get the next request
        request = self.prefill_queue.popleft()

        if self.memory_planner and not self.memory_planner.has_allocation(request.req_id):
            if not self.can_accommodate_request(request):
                self.prefill_queue.appendleft(request)
                return
            self.memory_planner.allocate(request)
            self._refresh_memory_usage()
        else:
            # Check if we can accommodate this request
            if not self.can_accommodate_request(request):
                # Put it back in the queue for later
                self.prefill_queue.appendleft(request)
                return

        # Start prefill for this request
        self.current_prefill_request = request
        self.active_requests.add(request)
        request._prefill()  # Mark as in prefill phase

        # Emit prefill start event aligned to the current GPU timeline
        if self.event_callback:
            start_time = max(self.time_cursor, self.current_time)
            event = Event(
                timestamp=start_time,
                event_type=EventType.PREFILL_START,
                target=self.engine_id,
                data={'request_id': request.req_id},
                priority=EventPriority.HIGH
            )
            self.event_callback(event)

    def _process_individual_prefill(self, current_time: float = 0.0) -> bool:
        """Process prefill for an individual request (no batching)."""
        if not self.current_prefill_request:
            return False

        request = self.current_prefill_request

        # Calculate prefill time for this single request
        if self.pd_separation:
            prefill_duration = self._get_prefill_time(request.input_length, 1)
            first_token_decode_time = self._get_decode_time(request.input_length, 1)
            kv_transfer_time = self._get_kv_transfer_time(request)
            prefill_duration = max(prefill_duration, 0.0)
            first_token_decode_time = max(first_token_decode_time, 0.0)
            kv_transfer_time = max(kv_transfer_time, 0.0)
            prefill_duration_total = prefill_duration + first_token_decode_time + kv_transfer_time
        else:
            prefill_duration = self._get_prefill_time(request.input_length, 1)
            prefill_duration_total = prefill_duration

        # Ensure we have a reasonable minimum duration for visualization
        if prefill_duration_total <= 0:
            prefill_duration_total = 0.001  # 1ms minimum

        # Align start time with current GPU timeline
        prefill_start_time = max(self.time_cursor, self.current_time)
        prefill_end_time = prefill_start_time + prefill_duration_total
        self.time_cursor = prefill_end_time

        # Create trace event for this prefill
        prefill_events = self.create_detailed_events(
            phase="prefill",
            handled_requests=[request],
            start_at=prefill_start_time,
            end_at=prefill_end_time
        )
        if self.pd_separation and prefill_events:
            # Attach PD-specific timing breakdown for visibility
            extra_args = prefill_events[0].args
            extra_args["pd_first_token_decode_s"] = first_token_decode_time
            extra_args["pd_kv_transfer_s"] = kv_transfer_time
            prefill_events[0].args = extra_args
        self.trace_events.extend(prefill_events)

        # Update statistics
        self.statistics['total_prefill_time'] += prefill_duration_total
        if self.pd_separation:
            # First token is produced on the prefill side
            request.generated_tokens = max(request.generated_tokens, 1)
            self.statistics['total_tokens_generated'] += 1
            # Prefill-only engine should drop active tracking after handoff
            if self.pd_prefill_only and request in self.active_requests:
                self.active_requests.remove(request)

        # Track prefill completions
        self.statistics['prefill_requests_processed'] += 1

        # Mark prefill as complete
        request.set_prefill_finished_at(prefill_end_time)

        # Emit prefill complete event
        if self.event_callback:
            event = Event(
                timestamp=prefill_end_time,
                event_type=EventType.PREFILL_COMPLETE,
                target=self.engine_id,
                data={
                    'request_id': request.req_id,
                    'prefill_time': prefill_duration_total
                },
                priority=EventPriority.HIGH
            )
            self.event_callback(event)

        return True

    def _form_decode_batch(self):
        """Form a decode batch from all ready requests."""
        if not self.decode_ready_requests or self.current_decode_batch is not None:
            return

        # Get all ready requests up to max_batch_size
        batch_requests = []
        while self.decode_ready_requests and len(batch_requests) < self.max_batch_size:
            candidate = self.decode_ready_requests[0]
            if self.can_accommodate_request(candidate):
                self.decode_ready_requests.popleft()
                batch_requests.append(candidate)
            else:
                # Not enough memory for the next in line; wait
                break

        # Create decode batch
        self.batch_counter += 1
        self.current_decode_batch = Batch(f"decode_batch_{self.batch_counter}", batch_requests)
        self.current_decode_batch.status = "decode"

        # Update statistics
        self.statistics['batch_sizes'].append(len(batch_requests))

    def _add_to_decode_batch(self):
        """Add new ready requests to existing decode batch (continuous batching)."""
        if not self.current_decode_batch or not self.decode_ready_requests:
            return

        current_batch_size = self.current_decode_batch.get_batch_size()
        while self.decode_ready_requests and self.current_decode_batch.get_batch_size() < self.max_batch_size:
            candidate = self.decode_ready_requests[0]
            if self.can_accommodate_request(candidate):
                self.decode_ready_requests.popleft()
                self.current_decode_batch.add_request(candidate)
            else:
                break

        # Update statistics to reflect the new batch size
        self.statistics['batch_sizes'].append(self.current_decode_batch.get_batch_size())

    def _process_decode_batch(self, current_time: float = 0.0) -> List[Event]:
        """Process one decode step for the current decode batch."""
        if not self.current_decode_batch:
            return []

        events = []
        completed_requests = []
        completed_request_ids: List[str] = []

        # Calculate decode time per token using ModelAnalyzer
        batch_size = len(self.current_decode_batch.requests)

        # Calculate current sequence length during decode (prompt + generated tokens)
        if self.current_decode_batch.requests:
            current_seq_length = max(
                req.input_length + req.generated_tokens for req in self.current_decode_batch.requests
            )
        else:
            current_seq_length = 1024  # fallback

        decode_duration = self._get_decode_time(current_seq_length, batch_size)

        # Ensure we have a reasonable minimum duration for visualization
        if decode_duration <= 0:
            decode_duration = 0.001  # 1ms minimum

        # Align decode step with current GPU timeline
        decode_start_time = max(self.time_cursor, self.current_time)
        decode_end_time = decode_start_time + decode_duration
        self.time_cursor = decode_end_time

        # Create detailed trace events for decode step
        # Only create events for requests that are still active
        active_requests = [req for req in self.current_decode_batch.requests if req.status != REQ_STATUS.EXIT]
        if active_requests:
            decode_events = self.create_detailed_events(
                phase="decode",
                handled_requests=active_requests,
                start_at=decode_start_time,
                end_at=decode_end_time
            )
            self.trace_events.extend(decode_events)

        # Process decode step for each request
        for request in self.current_decode_batch.requests:
            is_complete = request._decode()
            if self.memory_planner:
                self.memory_planner.allocate(request)
            if is_complete:
                completed_requests.append(request)
                completed_request_ids.append(request.req_id)

        # Update statistics
        self.statistics['total_decode_time'] += decode_duration
        self.statistics['total_tokens_generated'] += batch_size

        # Update KV cache memory
        self._update_kv_cache_memory()

        # Handle completed requests
        for request in completed_requests:
            self.active_requests.remove(request)
            self.completed_requests.append(request)
            self.current_decode_batch.remove_request(request)
            self.statistics['requests_processed'] += 1

            events.append(Event(
                timestamp=decode_end_time,
                event_type=EventType.REQUEST_COMPLETE,
                target=self.engine_id,
                data={
                    'request_id': request.req_id,
                    'total_tokens': request.generated_tokens,
                    'completion_time': decode_end_time
                },
                priority=EventPriority.HIGH
            ))

        if self.memory_planner and completed_request_ids:
            self.memory_planner.free(completed_request_ids)
            self._refresh_memory_usage()

        return events

    
    def _estimate_kv_cache_memory(self, request: GenerationRequest) -> float:
        """Estimate KV cache memory requirement for a request."""
        # Simplified estimation - would use actual model parameters
        hidden_size = getattr(self.model_params, 'hidden_size', 4096)
        num_layers = getattr(self.model_params, 'num_hidden_layers', 32)
        num_heads = getattr(self.model_params, 'num_attention_heads', 32)
        head_dim = hidden_size // num_heads  # Usually 128 for LLaMA models

        # KV cache per token: 2 * num_layers * num_heads * head_dim * bytes_per_element
        # The 2 is for Key and Value matrices
        bytes_per_element = 2  # FP16
        kv_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_element

        # Estimate maximum sequence length during generation
        max_seq_length = request.input_length + request.output_length
        total_kv_memory = kv_per_token * max_seq_length

        return total_kv_memory

    def _estimate_kv_cache_memory_for_length(self, sequence_length: int) -> float:
        """Estimate KV cache memory for a specific sequence length."""
        hidden_size = getattr(self.model_params, 'hidden_size', 4096)
        num_layers = getattr(self.model_params, 'num_hidden_layers', 32)
        num_heads = getattr(self.model_params, 'num_attention_heads', 32)
        head_dim = hidden_size // num_heads
        bytes_per_element = 2  # FP16
        kv_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_element
        return kv_per_token * max(sequence_length, 1)

    def _get_kv_transfer_time(self, request: GenerationRequest) -> float:
        """Estimate KV cache transfer time using configured bandwidth."""
        kv_bytes = self._estimate_kv_cache_memory_for_length(request.input_length)
        bandwidth_bytes_per_s = max(self.kv_transfer_bandwidth_gbps, 1e-6) * 1e9
        return kv_bytes / bandwidth_bytes_per_s

    def _update_kv_cache_memory(self):
        """Update total KV cache memory usage."""
        if self.memory_planner:
            self._refresh_memory_usage()
            return

        total_kv_memory = 0.0

        # Include active requests (currently in prefill or decode batch)
        for request in self.active_requests:
            current_seq_length = request.input_length + request.generated_tokens
            # Simplified calculation
            hidden_size = getattr(self.model_params, 'hidden_size', 4096)
            num_layers = getattr(self.model_params, 'num_hidden_layers', 32)
            num_heads = getattr(self.model_params, 'num_attention_heads', 32)
            head_dim = hidden_size // num_heads
            bytes_per_element = 2

            kv_per_token = 2 * head_dim * num_layers * num_heads * bytes_per_element
            total_kv_memory += kv_per_token * current_seq_length

        # Include decode-ready requests (completed prefill, waiting for decode batch)
        for request in self.decode_ready_requests:
            current_seq_length = request.input_length + request.generated_tokens
            hidden_size = getattr(self.model_params, 'hidden_size', 4096)
            num_layers = getattr(self.model_params, 'num_hidden_layers', 32)
            num_heads = getattr(self.model_params, 'num_attention_heads', 32)
            head_dim = hidden_size // num_heads
            bytes_per_element = 2

            kv_per_token = 2 * head_dim * num_layers * num_heads * bytes_per_element
            total_kv_memory += kv_per_token * current_seq_length

        self.kv_cache_memory = total_kv_memory
        self.statistics['memory_peak'] = max(self.statistics['memory_peak'],
                                            self.model_memory_usage + self.kv_cache_memory)

    def release_request_allocation(self, request_id: str):
        """Release KV allocation for a request if block-accounting is enabled."""
        if not self.memory_planner:
            return
        if self.memory_planner.has_allocation(request_id):
            self.memory_planner.free([request_id])
            self._refresh_memory_usage()

    def _get_prefill_time(self, sequence_length: int, batch_size: int) -> float:
        """Get accurate prefill time using ModelAnalyzer."""
        try:
            results = self.model_analyzer.analyze(
                seqlen=sequence_length,
                batchsize=batch_size,
                w_bit=16,  # Default to 16-bit weights
                a_bit=16,  # Default to 16-bit activations
                kv_bit=16,  # Default to 16-bit KV cache
                disable_attention=self.disable_attention,
                disable_ffn=self.disable_ffn
            )
            return results["total_results"]["prefill"]["inference_time"]
        except Exception as e:
            # Fallback to simplified estimate if ModelAnalyzer fails
            import traceback
            print(f"Warning: ModelAnalyzer prefill failed, using fallback: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            attention_ops = sequence_length * sequence_length * batch_size
            ops_per_second = self.compute_throughput
            if ops_per_second <= 0:
                return 0  # Avoid division by zero
            return attention_ops / ops_per_second

    def _get_decode_time(self, sequence_length: int, batch_size: int) -> float:
        """Get accurate decode time using ModelAnalyzer."""
        try:
            results = self.model_analyzer.analyze(
                seqlen=sequence_length,
                batchsize=batch_size,
                w_bit=16,  # Default to 16-bit weights
                a_bit=16,  # Default to 16-bit activations
                kv_bit=16,  # Default to 16-bit KV cache
                disable_attention=self.disable_attention,
                disable_ffn=self.disable_ffn
            )
            return results["total_results"]["decode"]["inference_time"]
        except Exception as e:
            # Fallback to simplified estimate if ModelAnalyzer fails
            import traceback
            print(f"Warning: ModelAnalyzer decode failed, using fallback: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            avg_seq_length = 1024  # Placeholder
            attention_ops = avg_seq_length * batch_size
            ops_per_second = self.compute_throughput
            if ops_per_second <= 0:
                return 0  # Avoid division by zero
            return attention_ops / ops_per_second

    def set_current_time(self, current_time: float):
        """Set the current simulation time for tracing purposes."""
        self.current_time = current_time
        if self.time_cursor < current_time:
            self.time_cursor = current_time

    def add_trace_event(self, name: str, category: str, phase: str,
                       timestamp: float, duration: Optional[float] = None,
                       args: Optional[Dict[str, Any]] = None):
        """Add a trace event for Chrome tracing visualization."""
        trace_event = TraceEvent(
            name=name,
            cat=category,
            ph=phase,
            pid=self.engine_id,
            tid="engine",
            ts=int(timestamp * 1e6),  # Convert to microseconds
            args=args or {},
            dur=int(duration * 1e6) if duration is not None else None
        )
        self.trace_events.append(trace_event)

    def create_detailed_events(self, phase: str, handled_requests: List[GenerationRequest],
                              start_at: float, end_at: float) -> List[TraceEvent]:
        """
        Create Chrome trace format events for detailed performance visualization.

        Args:
            phase: Either "prefill" or "decode"
            handled_requests: List of requests processed in this phase
            start_at: Start time in seconds
            end_at: End time in seconds

        Returns:
            List of TraceEvent objects compatible with Chrome tracing format
        """
        complete_events = []
        start_us = int(max(start_at, 0) * 1_000_000)
        duration_s = max(end_at - start_at, 0.0)
        duration_us = max(int(duration_s * 1_000_000), 1)

        for req in handled_requests:
            event_args = {
                "request_id": req.req_id,
                "requested_model": req.model,
                "engine_id": str(self.engine_id),
                "engine_model": self.model_id,
                "hardware": self.hardware,
                "phase": phase,
                "start_time_s": round(start_at, 6),
                "end_time_s": round(end_at, 6),
                "duration_s": round(duration_s, 6),
            }

            if phase == "prefill":
                event_args.update(
                    {
                        "prompt_tokens": req.input_length,
                        "target_output_tokens": req.output_length,
                    }
                )
            elif phase == "decode":
                event_args.update(
                    {
                        "target_output_tokens": req.output_length,
                        "generated_tokens_total": req.generated_tokens,
                        "tokens_emitted_this_step": 1,
                    }
                )

            complete_events.append(
                TraceEvent(
                    name=f"{phase.upper()[0]}:{req.req_id}",
                    cat=f"request.{phase}",
                    ph="X",  # Complete event (duration event)
                    pid=str(self.engine_id),
                    tid=0,   # Single thread for the engine
                    ts=start_us,
                    dur=duration_us,
                    args=event_args,
                )
            )

        return complete_events

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        avg_batch_size = np.mean(self.statistics['batch_sizes']) if self.statistics['batch_sizes'] else 0

        return {
            **self.statistics,
            'engine_id': self.engine_id,
            'model_id': self.model_id,
            'hardware': self.hardware,
            'current_load': len(self.active_requests) + len(self.decode_ready_requests) + (1 if self.current_prefill_request else 0),
            'queue_length': len(self.prefill_queue),
            'prefill_queue_length': len(self.prefill_queue),
            'decode_ready_count': len(self.decode_ready_requests),
            'current_prefill_request': self.current_prefill_request is not None,
            'current_decode_batch_size': len(self.current_decode_batch.requests) if self.current_decode_batch else 0,
            'avg_batch_size': avg_batch_size,
            'memory_info': self.get_memory_info(),
            'model_loaded': self.model_loaded,
            'trace_events': self.trace_events
        }
