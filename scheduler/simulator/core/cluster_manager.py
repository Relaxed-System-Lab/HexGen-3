from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import deque
import numpy as np
from .events import EventLoop, Event, EventType, EventPriority
from .engine import ServingEngine
from .request import GenerationRequest
from .arrival import ArrivalProcess
from .scheduler import get_scheduler
from .trace import TraceEvent, get_request_color
from .afd_analyzer import AFDAnalyzer


@dataclass
class NodeConfiguration:
    """Configuration for a cluster node."""
    node_id: str
    model_id: str
    hardware: str
    max_batch_size: Optional[int] = None
    parallel_config: Optional[Any] = None
    disable_attention: bool = False
    disable_ffn: bool = False
    pd_separation: bool = False
    pd_prefill_only: bool = False
    pd_decode_only: bool = False
    kv_transfer_bandwidth_gbps: float = 100.0
    afd_attention: bool = False
    afd_ffn: bool = False
    afd_activation_bandwidth_gbps: float = 100.0
    afd_enabled: bool = False


@dataclass
class ClusterConfiguration:
    """Configuration for the entire cluster."""
    cluster_id: str
    nodes: List[NodeConfiguration]
    scheduler_algorithm: str
    scheduler_config: Optional[Dict[str, Any]] = None
    afd_enabled: bool = False
    afd_attention_batch_size: int = 6
    afd_ffn_max_batch_size: int = 6
    afd_activation_bandwidth_gbps: float = 100.0


class ClusterManager:
    """Manages multiple serving engines and orchestrates cluster-wide scheduling."""

    def __init__(self, config: ClusterConfiguration, arrival_process: ArrivalProcess):
        """
        Initialize the cluster manager.

        Args:
            config: Cluster configuration
            arrival_process: Process for generating request arrivals
        """
        self.config = config
        self.arrival_process = arrival_process

        # Event loop for cluster coordination
        self.event_loop = EventLoop()

        # Node management
        self.serving_engines: Dict[str, ServingEngine] = {}
        self.online_nodes: Set[str] = set()
        self.offline_nodes: Set[str] = set()

        # Scheduling
        self.scheduler = get_scheduler(
            config.scheduler_algorithm,
            **(config.scheduler_config or {})
        )

        # Request tracking
        self.active_requests: Dict[str, GenerationRequest] = {}
        self.completed_requests: List[GenerationRequest] = []
        self.rejected_requests: List[Dict[str, Any]] = []
        self.request_counter = 0

        # Performance monitoring
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'rejected_requests': 0,
            'cluster_utilization': 0.0,
            'average_latency': 0.0,
            'throughput': 0.0,
            'scheduler_decisions': {},
            'node_failures': 0,
            'node_recoveries': 0
        }

        # Timing
        self.start_time = 0.0
        self.current_time = 0.0
        self.last_metrics_update = 0.0
        self.metrics_update_interval = 1.0  # Update metrics every 1 second

        # Trace events for analysis
        self.trace_events: List[TraceEvent] = []

        # AFD state
        self.afd_enabled = self.config.afd_enabled
        self.afd_attention_nodes: List[str] = []
        self.afd_ffn_nodes: List[str] = []
        self.afd_waitlist: deque[GenerationRequest] = deque()
        self.afd_analyzer: Optional[AFDAnalyzer] = None
        self.afd_next_pull_at: Optional[float] = None
        self.afd_requests_processed: int = 0
        self.afd_tokens_generated: int = 0
        self.afd_request_components: Dict[str, Dict[str, float]] = {}

        # Initialize cluster
        self._initialize_cluster()

    def _initialize_cluster(self):
        """Initialize cluster nodes and register event handlers."""
        # Create serving engines for each node
        for node_config in self.config.nodes:
            engine = ServingEngine(
                engine_id=node_config.node_id,
                model_id=node_config.model_id,
                model_instance=None,  # Will be loaded as needed
                hardware=node_config.hardware,
                parallel_config=node_config.parallel_config,
                max_batch_size=node_config.max_batch_size,
                disable_attention=node_config.disable_attention,
                disable_ffn=node_config.disable_ffn,
                pd_separation=node_config.pd_separation,
                pd_prefill_only=node_config.pd_prefill_only,
                pd_decode_only=node_config.pd_decode_only,
                kv_transfer_bandwidth_gbps=node_config.kv_transfer_bandwidth_gbps,
                afd_enabled=self.config.afd_enabled,
                afd_attention=node_config.afd_attention,
                afd_ffn=node_config.afd_ffn,
            )

            # Set event callback for engine-to-cluster communication
            engine.set_event_callback(self._handle_engine_event)

            self.serving_engines[node_config.node_id] = engine
            self.online_nodes.add(node_config.node_id)
            if node_config.afd_attention:
                self.afd_attention_nodes.append(node_config.node_id)
            if node_config.afd_ffn:
                self.afd_ffn_nodes.append(node_config.node_id)

        if self.afd_enabled:
            if not self.afd_attention_nodes or not self.afd_ffn_nodes:
                raise ValueError("AFD mode requires at least one attention node and one FFN node")

            # Use per-instance analyzers so heterogeneous AFD attention/FFN
            # placements keep their hardware-specific timing.
            attn_engine = self.serving_engines[self.afd_attention_nodes[0]]
            self.afd_analyzer = AFDAnalyzer(
                model_analyzer=attn_engine.model_analyzer,
                attention_batch_size=self.config.afd_attention_batch_size,
                ffn_max_batch_size=self.config.afd_ffn_max_batch_size,
                num_attention_instances=len(self.afd_attention_nodes),
                num_ffn_instances=len(self.afd_ffn_nodes),
                kv_transfer_bandwidth_gbps=attn_engine.kv_transfer_bandwidth_gbps,
                activation_bandwidth_gbps=self.config.afd_activation_bandwidth_gbps,
                attention_analyzers=[
                    self.serving_engines[node_id].model_analyzer
                    for node_id in self.afd_attention_nodes
                ],
                ffn_analyzers=[
                    self.serving_engines[node_id].model_analyzer
                    for node_id in self.afd_ffn_nodes
                ],
            )
            self.afd_analyzer.reset()

        # Register cluster event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register event handlers for cluster-level events."""
        self.event_loop.register_handler(EventType.REQUEST_ARRIVAL, self._handle_request_arrival)
        self.event_loop.register_handler(EventType.PLACEMENT_DECISION, self._handle_placement_decision)
        self.event_loop.register_handler(EventType.NODE_ONLINE, self._handle_node_online)
        self.event_loop.register_handler(EventType.NODE_OFFLINE, self._handle_node_offline)
        self.event_loop.register_handler(EventType.REQUEST_COMPLETE, self._handle_request_complete)
        self.event_loop.register_handler(EventType.LOAD_BALANCE, self._handle_load_balance)

        # Register local engine event handlers
        self.event_loop.register_handler(EventType.PREFILL_START, self._handle_prefill_start)
        self.event_loop.register_handler(EventType.PREFILL_COMPLETE, self._handle_prefill_complete)
        self.event_loop.register_handler(EventType.DECODE_STEP, self._handle_decode_step)
        self.event_loop.register_handler(EventType.BATCH_FORM, self._handle_batch_form)
        self.event_loop.register_handler(EventType.MEMORY_CHECK, self._handle_memory_check)
        self.event_loop.register_handler(EventType.MODEL_LOAD, self._handle_model_load)
        self.event_loop.register_handler(EventType.MODEL_UNLOAD, self._handle_model_unload)
        self.event_loop.register_handler(EventType.AFD_ATTENTION_PULL, self._handle_afd_pull)

    def _handle_engine_event(self, event: Event):
        """Handle events from serving engines."""
        # Set timestamp and schedule in cluster event loop
        event.timestamp = self.event_loop.current_time
        self.event_loop.schedule_event(event)

    def _handle_request_arrival(self, event: Event):
        """Handle new request arrival."""
        request_data = event.data
        request = GenerationRequest(
            req_id=request_data['request_id'],
            model=request_data['model'],
            input_length=request_data['input_length'],
            output_length=request_data['output_length'],
            arrive_at=event.timestamp
        )

        self.active_requests[request.req_id] = request
        self.metrics['total_requests'] += 1

        # Add trace event for request arrival (start of request lifecycle)
        trace_event = TraceEvent(
            name="RequestArrival",
            cat="request_lifecycle",
            ph="B",  # Begin event
            pid="request_tracker",  # New process for overall request tracking
            tid="lifecycle",  # Single thread for all requests
            ts=int(event.timestamp * 1e6),  # Convert to microseconds
            cname=get_request_color(request.req_id),  # Color for this request
            args={
                'request_id': request.req_id,
                'model': request.model,
                'input_length': request.input_length,
                'output_length': request.output_length
            }
        )
        self.trace_events.append(trace_event)

        # Schedule placement decision
        placement_event = Event(
            timestamp=event.timestamp,
            event_type=EventType.PLACEMENT_DECISION,
            target="cluster_manager",
            data={'request_id': request.req_id, 'phase': 'prefill'},
            priority=EventPriority.MEDIUM
        )
        self.event_loop.schedule_event(placement_event)

    def _handle_placement_decision(self, event: Event):
        """Make placement decision for a request."""
        request_id = event.data['request_id']
        request = self.active_requests.get(request_id)
        if not request:
            return

        phase = event.data.get('phase', 'prefill')
        pd_mode = any(
            engine.pd_separation
            for node_id, engine in self.serving_engines.items()
            if node_id in self.online_nodes
        )

        # Get available engines
        if phase == 'prefill':
            if pd_mode:
                available_engines = [
                    engine for node_id, engine in self.serving_engines.items()
                    if node_id in self.online_nodes and engine.pd_prefill_only
                ]
            else:
                # Non-PD mode: any online engine can handle both prefill and decode.
                available_engines = [
                    engine for node_id, engine in self.serving_engines.items()
                    if node_id in self.online_nodes
                ]
            if not available_engines:
                self._reject_request(request, "no_prefill_engine_available")
                return
        else:  # decode placement
            if self.afd_enabled:
                self._cleanup_prefill_residuals(request)
                self._enqueue_afd_request(request)
                return
            if pd_mode:
                available_engines = [
                    engine for node_id, engine in self.serving_engines.items()
                    if node_id in self.online_nodes and engine.pd_decode_only
                ]
            else:
                available_engines = [
                    engine for node_id, engine in self.serving_engines.items()
                    if node_id in self.online_nodes
                ]
            if not available_engines:
                self._reject_request(request, "no_decode_engine_available")
                return

        # Make placement decision
        decision = self.scheduler.place_request(request, available_engines)

        if decision:
            # Place request on selected engine
            if phase == 'prefill':
                success = decision.target_engine.add_request(request)
                if success:
                    request.prefill_node = decision.target_engine.engine_id
            else:
                success = decision.target_engine.add_decode_ready_request(
                    request, ready_time=self.event_loop.current_time
                )
                if success:
                    self._cleanup_prefill_residuals(request)
                    request.decode_node = decision.target_engine.engine_id
                    request.serving_node = decision.target_engine.engine_id

            if success:
                self._record_placement_decision(decision)

                # Add trace event for successful placement
                placement_trace_event = TraceEvent(
                    name="RequestScheduled",
                    cat="request_lifecycle",
                    ph="I",  # Instant event
                    pid="request_tracker",
                    tid="lifecycle",  # Single thread for all requests
                    ts=int(self.event_loop.current_time * 1e6),
                    cname=get_request_color(request_id),  # Same color as arrival
                    args={
                        'request_id': request_id,
                        'target_engine': decision.target_engine.engine_id,
                        'placement_reason': decision.reason,
                        'phase': phase,
                        'estimated_latency': getattr(decision, 'estimated_latency', None)
                    }
                )
                self.trace_events.append(placement_trace_event)

                # Trigger engine step
                self._schedule_engine_step(decision.target_engine.engine_id)
            else:
                self._reject_request(request, "engine_rejection")
        else:
            self._reject_request(request, "no_suitable_engine")

    def _handle_request_complete(self, event: Event):
        """Handle request completion."""
        request_id = event.data['request_id']
        request = self.active_requests.get(request_id)

        if request:
            request.set_generation_finished_at(event.timestamp)
            self.completed_requests.append(request)
            del self.active_requests[request_id]

            self.metrics['completed_requests'] += 1

            # Add trace event for request completion (end of request lifecycle)
            trace_event = TraceEvent(
                name="RequestComplete",
                cat="request_lifecycle",
                ph="E",  # End event
                pid="request_tracker",  # Same process as arrival for proper pairing
                tid="lifecycle",  # Single thread for all requests
                ts=int(event.timestamp * 1e6),  # Convert to microseconds
                cname=get_request_color(request_id),  # Same color as arrival
                args={
                    'request_id': request_id,
                    'total_time': event.timestamp - request.arrive_at,
                    'tokens_generated': request.generated_tokens
                }
            )
            self.trace_events.append(trace_event)

    def _handle_node_online(self, event: Event):
        """Handle node coming online."""
        node_id = event.data['node_id']
        if node_id in self.offline_nodes:
            self.offline_nodes.remove(node_id)
            self.online_nodes.add(node_id)
            self.metrics['node_recoveries'] += 1

            print(f"Node {node_id} is back online")

    def _handle_node_offline(self, event: Event):
        """Handle node going offline."""
        node_id = event.data['node_id']
        if node_id in self.online_nodes:
            self.online_nodes.remove(node_id)
            self.offline_nodes.add(node_id)
            self.metrics['node_failures'] += 1

            # Move pending requests to other nodes
            self._rebalance_requests(node_id)

            print(f"Node {node_id} went offline")

    def _handle_load_balance(self, event: Event):
        """Handle load balancing across nodes."""
        # Simple load balancing - can be extended with more sophisticated algorithms
        self._rebalance_requests()

    # Local engine event handlers
    def _handle_prefill_start(self, event: Event):
        """Handle prefill start event from an engine."""
        # Can be used for logging, monitoring, or additional coordination
        pass

    def _handle_prefill_complete(self, event: Event):
        """Handle prefill complete event from an engine."""
        request_id = event.data.get('request_id')
        if not request_id:
            return
        engine = self.serving_engines.get(event.target)
        if not engine:
            return

        request = self.active_requests.get(request_id)
        if not request:
            return

        if self.afd_enabled:
            self._cleanup_prefill_residuals(request)
            self._enqueue_afd_request(request)
            return

        if not engine.pd_separation:
            return

        # Dispatch to decode placement
        placement_event = Event(
            timestamp=event.timestamp,
            event_type=EventType.PLACEMENT_DECISION,
            target="cluster_manager",
            data={'request_id': request_id, 'phase': 'decode'},
            priority=EventPriority.MEDIUM
        )
        self.event_loop.schedule_event(placement_event)

    def _handle_decode_step(self, event: Event):
        """Handle decode step event from an engine."""
        # Can be used for logging, monitoring, or additional coordination
        pass

    def _enqueue_afd_request(self, request: GenerationRequest):
        """Add a decode-ready request into the global AFD queue and try to schedule it."""
        if not self.afd_enabled or not self.afd_analyzer:
            return

        self.afd_waitlist.append(request)

        trace_event = TraceEvent(
            name="AFDEnqueue",
            cat="afd",
            ph="i",
            pid="request_tracker",
            tid="afd_queue",
            ts=int(self.event_loop.current_time * 1e6),
            cname=get_request_color(request.req_id),
            args={
                "request_id": request.req_id,
                "queue_length": len(self.afd_waitlist),
            },
        )
        self.trace_events.append(trace_event)
        self._schedule_next_afd_pull(self.event_loop.current_time)

    def _schedule_next_afd_pull(self, not_before: float):
        """Schedule the next attention pull event if needed."""
        if not self.afd_enabled or not self.afd_analyzer or not self.afd_waitlist:
            return

        next_available = self.afd_analyzer.next_attention_available_time()
        t = max(not_before, next_available, self.event_loop.current_time)
        if self.afd_next_pull_at is not None and t >= self.afd_next_pull_at:
            return

        pull_event = Event(
            timestamp=t,
            event_type=EventType.AFD_ATTENTION_PULL,
            target="cluster_manager",
            data={},
            priority=EventPriority.HIGH,
        )
        self.afd_next_pull_at = t
        self.event_loop.schedule_event(pull_event)

    def _handle_afd_pull(self, event: Event):
        """Handle attention instance pulling a batch from the global queue."""
        self.afd_next_pull_at = None
        if not self.afd_enabled or not self.afd_analyzer:
            return
        if not self.afd_waitlist:
            return

        attention_nodes = self.afd_attention_nodes
        result = self.afd_analyzer.schedule_next_batch(
            waitlist=self.afd_waitlist,
            start_time=event.timestamp,
        )
        if not result:
            return

        batch_report = result["batch_report"]
        attn_idx = batch_report.get("attention_instance", 0)
        node_id = attention_nodes[attn_idx] if attn_idx < len(attention_nodes) else None

        max_output_len = max(batch_report.get("max_output_len", 1), 1)
        num_layers = batch_report.get("num_layers", 1)
        attn_total = batch_report.get("attention_layer_time_s", 0.0) * num_layers * max_output_len
        ffn_total = (batch_report.get("ffn_layer_time_s", 0.0) + batch_report.get("ffn_wait_s", 0.0)) * num_layers * max_output_len

        for req in result["batch"]:
            req.decode_node = node_id
            req.serving_node = node_id
            req.generated_tokens = req.output_length
            self.afd_requests_processed += 1
            self.afd_tokens_generated += req.output_length
            share = req.output_length / max_output_len if max_output_len > 0 else 1.0 / max(len(result["batch"]), 1)
            self.afd_request_components[req.req_id] = {
                "attn_s": attn_total * share,
                "ffn_s": ffn_total * share,
            }

        for req_id, completion_time in result["completion_times"].items():
            complete_event = Event(
                timestamp=completion_time,
                event_type=EventType.REQUEST_COMPLETE,
                target="cluster_manager",
                data={"request_id": req_id},
                priority=EventPriority.HIGH,
            )
            self.event_loop.schedule_event(complete_event)

        # Schedule the next pull if there is still work
        if self.afd_waitlist:
            self._schedule_next_afd_pull(not_before=event.timestamp)

    def _handle_batch_form(self, event: Event):
        """Handle batch formation event from an engine."""
        # Process the engine step and get resulting events
        engine_id = event.target
        engine = self.serving_engines.get(engine_id)
        if engine:
            # Process engine step with the current simulation time
            engine_events = engine.step(self.event_loop.current_time)

            # Schedule events generated by the engine using their absolute timestamps
            for engine_event in engine_events:
                if engine_event.timestamp < self.event_loop.current_time:
                    # Guard against numerical drift causing past scheduling
                    engine_event.timestamp = self.event_loop.current_time
                self.event_loop.schedule_event(engine_event)

            # Schedule next BATCH_FORM event if engine still has work pending
            has_pending_work = (
                engine.prefill_queue or engine.current_prefill_request or
                engine.decode_ready_requests or
                (engine.current_decode_batch and not engine.current_decode_batch.is_empty())
            )

            if has_pending_work:
                next_step_time = max(engine.time_cursor, self.event_loop.current_time)
                if next_step_time <= self.event_loop.current_time:
                    # Ensure progress even if durations are extremely small
                    next_step_time = self.event_loop.current_time + 1e-6

                step_event = Event(
                    timestamp=next_step_time,
                    event_type=EventType.BATCH_FORM,
                    target=engine_id,
                    data={},
                    priority=EventPriority.HIGH
                )
                self.event_loop.schedule_event(step_event)

    def _handle_memory_check(self, event: Event):
        """Handle memory check event from an engine."""
        # Can be used for logging, monitoring, or additional coordination
        pass

    def _handle_model_load(self, event: Event):
        """Handle model load event from an engine."""
        # Can be used for logging, monitoring, or additional coordination
        pass

    def _handle_model_unload(self, event: Event):
        """Handle model unload event from an engine."""
        # Can be used for logging, monitoring, or additional coordination
        pass

    def _schedule_engine_step(self, engine_id: str):
        """Schedule a step event for a specific engine."""
        step_event = Event(
            timestamp=self.event_loop.current_time,
            event_type=EventType.BATCH_FORM,
            target=engine_id,
            data={},
            priority=EventPriority.HIGH
        )
        self.event_loop.schedule_event(step_event)

    def _reject_request(self, request: GenerationRequest, reason: str):
        """Reject a request and record metrics."""
        self.rejected_requests.append({
            'req_id': request.req_id,
            'model': request.model,
            'input_length': request.input_length,
            'output_length': request.output_length,
            'arrive_at': request.arrive_at,
            'prefill_node': request.prefill_node,
            'decode_node': request.decode_node,
            'reason': reason
        })
        del self.active_requests[request.req_id]
        self.metrics['rejected_requests'] += 1

        trace_event = TraceEvent(
            name="RequestRejected",
            cat="request_lifecycle",
            ph="E",
            pid="request_tracker",  # Same process for consistency
            tid="lifecycle",  # Single thread for all requests
            ts=int(self.event_loop.current_time * 1e6),
            cname=get_request_color(request.req_id),  # Same color as arrival
            args={'reason': reason}
        )
        self.trace_events.append(trace_event)

    def _record_placement_decision(self, decision):
        """Record placement decision for metrics."""
        reason = decision.reason
        self.metrics['scheduler_decisions'][reason] = \
            self.metrics['scheduler_decisions'].get(reason, 0) + 1

    def _cleanup_prefill_residuals(self, request: GenerationRequest):
        """Remove request from prefill engine tracking to avoid double counting."""
        prefill_engine_id = request.prefill_node
        if not prefill_engine_id:
            return
        prefill_engine = self.serving_engines.get(prefill_engine_id)
        if not prefill_engine:
            return
        # Remove from active sets and queues if still present
        if request in prefill_engine.active_requests:
            prefill_engine.active_requests.remove(request)
        if request in prefill_engine.prefill_queue:
            try:
                prefill_engine.prefill_queue.remove(request)
            except ValueError:
                pass
        if prefill_engine.current_prefill_request == request:
            prefill_engine.current_prefill_request = None
        prefill_engine.release_request_allocation(request.req_id)

    def _rebalance_requests(self, failed_node_id: Optional[str] = None):
        """Rebalance requests after node failure or for load balancing."""
        if failed_node_id:
            # Handle node-specific rebalancing
            engine = self.serving_engines.get(failed_node_id)
            if engine:
                # Move queued requests to other nodes
                queued_requests = list(engine.request_queue)
                engine.request_queue.clear()

                for request in queued_requests:
                    # Re-queue for placement decision
                    placement_event = Event(
                        timestamp=self.event_loop.current_time,
                        event_type=EventType.PLACEMENT_DECISION,
                        target="cluster_manager",
                        data={'request_id': request.req_id, 'phase': 'prefill'},
                        priority=EventPriority.MEDIUM
                    )
                    self.event_loop.schedule_event(placement_event)

    def _update_metrics(self):
        """Update cluster-wide performance metrics."""
        if self.event_loop.current_time - self.last_metrics_update < self.metrics_update_interval:
            return

        # Calculate cluster utilization
        total_gpu_memory = sum(
            engine.hardware_spec['vmemory']
            for engine in self.serving_engines.values()
        )
        used_memory = sum(
            engine.get_memory_info()['used']
            for engine in self.serving_engines.values()
        )
        self.metrics['cluster_utilization'] = used_memory / total_gpu_memory if total_gpu_memory > 0 else 0

        # Calculate average latency
        if self.completed_requests:
            latencies = [
                req.generation_finished_at - req.arrive_at
                for req in self.completed_requests
                if req.generation_finished_at is not None
            ]
            self.metrics['average_latency'] = np.mean(latencies) if latencies else 0

        # Calculate throughput
        elapsed_time = self.event_loop.current_time - self.start_time
        if elapsed_time > 0:
            self.metrics['throughput'] = len(self.completed_requests) / elapsed_time

        self.last_metrics_update = self.event_loop.current_time

    def schedule_request_arrivals(self, duration: float):
        """Schedule request arrivals for the simulation duration."""
        arrival_times = self.arrival_process.generate_arrivals(
            start=self.start_time,
            duration=duration
        )

        for arrival_time in arrival_times:
            self.request_counter += 1
            request_id = f"req_{self.request_counter}"

            # Generate request parameters (simplified - could be more sophisticated)
            input_length = np.random.randint(128, 2048)  # 128-2048 tokens
            output_length = np.random.randint(32, 512)   # 32-512 tokens

            arrival_event = Event(
                timestamp=arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                target="cluster_manager",
                data={
                    'request_id': request_id,
                    'model': self.config.nodes[0].model_id,  # Simplified - single model
                    'input_length': input_length,
                    'output_length': output_length
                },
                priority=EventPriority.MEDIUM
            )
            self.event_loop.schedule_event(arrival_event)

    def simulate_node_failure(self, node_id: str, failure_time: float, recovery_time: float):
        """Simulate node failure and recovery."""
        # Schedule node failure
        failure_event = Event(
            timestamp=failure_time,
            event_type=EventType.NODE_OFFLINE,
            target="cluster_manager",
            data={'node_id': node_id},
            priority=EventPriority.HIGHEST
        )
        self.event_loop.schedule_event(failure_event)

        # Schedule node recovery
        recovery_event = Event(
            timestamp=recovery_time,
            event_type=EventType.NODE_ONLINE,
            target="cluster_manager",
            data={'node_id': node_id},
            priority=EventPriority.HIGHEST
        )
        self.event_loop.schedule_event(recovery_event)

    def run_simulation(self, duration: float, enable_failures: bool = False):
        """Run the cluster simulation."""
        print(f"Starting cluster simulation for {duration} seconds...")
        print(f"Cluster ID: {self.config.cluster_id}")
        print(f"Nodes: {len(self.config.nodes)}")
        print(f"Scheduler: {self.config.scheduler_algorithm}")
        print(f"Arrival process: {self.arrival_process}")

        self.start_time = 0.0
        self.current_time = 0.0

        # Schedule request arrivals
        self.schedule_request_arrivals(duration)

        # Schedule periodic load balancing
        balance_interval = 10.0  # Every 10 seconds
        current_balance_time = balance_interval
        while current_balance_time < duration:
            balance_event = Event(
                timestamp=current_balance_time,
                event_type=EventType.LOAD_BALANCE,
                target="cluster_manager",
                data={},
                priority=EventPriority.LOW
            )
            self.event_loop.schedule_event(balance_event)
            current_balance_time += balance_interval

        # Schedule optional node failures
        if enable_failures and len(self.config.nodes) > 1:
            # Simulate failure of first node at 30% of simulation time
            failed_node = self.config.nodes[0].node_id
            failure_time = duration * 0.3
            recovery_time = duration * 0.5
            self.simulate_node_failure(failed_node, failure_time, recovery_time)

        # Run the event loop until all requests are complete
        # We use a larger time limit to ensure all requests can finish processing
        extended_time = duration * 10  # Allow up to 10x the duration for processing
        max_events = 100000  # Safety limit to prevent infinite loops
        print(f"Running simulation until all requests complete (max time: {extended_time}s, max events: {max_events})...")

        self.event_loop.run(max_time=extended_time, max_events=max_events)

        # Update the cluster manager's current time to match the event loop
        self.current_time = self.event_loop.current_time

        # If there are still active requests but no events, it means the engines are stuck
        # This shouldn't happen in a well-functioning simulation
        if self.active_requests and not self.event_loop.event_queue:
            print(f"Warning: {len(self.active_requests)} requests still active but no events queued")
            print("This may indicate a problem with engine processing or event generation")

        # Final metrics update
        self._update_metrics()

        print("Simulation completed!")
        self.print_summary()

    def print_summary(self):
        """Print simulation summary."""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)

        print(f"Total requests: {self.metrics['total_requests']}")
        print(f"Completed requests: {self.metrics['completed_requests']}")
        print(f"Rejected requests: {self.metrics['rejected_requests']}")
        print(f"Success rate: {self.metrics['completed_requests'] / max(self.metrics['total_requests'], 1) * 100:.1f}%")
        print(f"Average latency: {self.metrics['average_latency']:.3f}s")
        print(f"Throughput: {self.metrics['throughput']:.2f} requests/sec")
        print(f"Cluster utilization: {self.metrics['cluster_utilization'] * 100:.1f}%")

        print(f"\nNode failures: {self.metrics['node_failures']}")
        print(f"Node recoveries: {self.metrics['node_recoveries']}")

        print(f"\nScheduler decisions:")
        for reason, count in self.metrics['scheduler_decisions'].items():
            print(f"  {reason}: {count}")

        if self.afd_enabled:
            print(f"\nAFD aggregate:")
            print(f"  Requests processed: {self.afd_requests_processed}")
            print(f"  Tokens generated: {self.afd_tokens_generated}")
            print(f"  Attention nodes: {self.afd_attention_nodes}")
            print(f"  FFN nodes: {self.afd_ffn_nodes}")
            print(f"\nEngine statistics (prefill nodes):")
            for engine_id, engine in self.serving_engines.items():
                if not engine.pd_prefill_only:
                    continue
                stats = engine.get_statistics()
                print(f"  {engine_id}:")
                print(f"    Prefill requests processed: {stats.get('prefill_requests_processed', 0)}")
                print(f"    Current load: {stats['current_load']}")
                print(f"    Memory utilization: {stats['memory_info']['utilization'] * 100:.1f}%")
        else:
            print(f"\nEngine statistics:")
            for engine_id, engine in self.serving_engines.items():
                stats = engine.get_statistics()
                print(f"  {engine_id}:")
                if engine.pd_prefill_only:
                    print(f"    Prefill requests processed: {stats.get('prefill_requests_processed', 0)}")
                else:
                    print(f"    Requests processed: {stats['requests_processed']}")
                print(f"    Tokens generated: {stats['total_tokens_generated']}")
                print(f"    Current load: {stats['current_load']}")
                print(f"    Memory utilization: {stats['memory_info']['utilization'] * 100:.1f}%")

        print(f"\nEvent loop statistics:")
        event_stats = self.event_loop.get_statistics()
        print(f"  Events processed: {event_stats['events_processed']}")
        print(f"  Simulation time: {event_stats['total_simulation_time']:.3f}s")

    def update_engine_timing(self, current_time: float):
        """Update current time on all engines for consistent tracing."""
        for engine_id, engine in self.serving_engines.items():
            engine.set_current_time(current_time)

    def set_node_pd_role(self, node_id: str, prefill_only: Optional[bool] = None, decode_only: Optional[bool] = None):
        """Dynamically change a node's PD role (for future reallocation)."""
        engine = self.serving_engines.get(node_id)
        if engine:
            engine.set_pd_role(prefill_only=prefill_only, decode_only=decode_only)

    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results."""
        engine_results = {}
        all_trace_events = list(self.trace_events)  # Start with cluster-level events

        # Update current time on all engines for consistent tracing
        current_time = self.event_loop.current_time if hasattr(self, 'event_loop') else self.current_time
        self.update_engine_timing(current_time)

        for engine_id, engine in self.serving_engines.items():
            engine_stats = engine.get_statistics()
            engine_results[engine_id] = engine_stats

            # Collect trace events from individual engines
            if 'trace_events' in engine_stats:
                all_trace_events.extend(engine_stats['trace_events'])

        return {
            'cluster_config': {
                'cluster_id': self.config.cluster_id,
                'scheduler': self.config.scheduler_algorithm,
                'num_nodes': len(self.config.nodes),
                'arrival_process': str(self.arrival_process)
            },
            'metrics': self.metrics,
            'afd_stats': {
                'enabled': self.afd_enabled,
                'attention_nodes': self.afd_attention_nodes,
                'ffn_nodes': self.afd_ffn_nodes,
                'requests_processed': self.afd_requests_processed,
                'tokens_generated': self.afd_tokens_generated,
            },
            'afd_request_components': self.afd_request_components if self.afd_enabled else {},
            'engines': engine_results,
            'completed_requests': [req.to_dict() for req in self.completed_requests],
            'rejected_requests': self.rejected_requests,
            'trace_events': all_trace_events,  # Combined trace events from cluster + engines
            'event_loop_stats': self.event_loop.get_statistics(),
            'scheduler_stats': self.scheduler.get_statistics()
        }
