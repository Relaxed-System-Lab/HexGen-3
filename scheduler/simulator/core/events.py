from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import heapq
import uuid


class EventType(Enum):
    """Event types for the cluster simulation."""
    # Cluster-level events
    REQUEST_ARRIVAL = "request_arrival"
    NODE_ONLINE = "node_online"
    NODE_OFFLINE = "node_offline"
    PLACEMENT_DECISION = "placement_decision"
    LOAD_BALANCE = "load_balance"

    # Local events (processed within ServingEngine)
    PREFILL_START = "prefill_start"
    PREFILL_COMPLETE = "prefill_complete"
    DECODE_STEP = "decode_step"
    BATCH_FORM = "batch_form"
    MEMORY_CHECK = "memory_check"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    REQUEST_COMPLETE = "request_complete"
    AFD_ATTENTION_PULL = "afd_attention_pull"


class EventPriority(Enum):
    """Event priorities for ordering events with same timestamp."""
    HIGHEST = 1  # Node failures, critical memory constraints
    HIGH = 2     # Request completions, batch formations
    MEDIUM = 3   # Placement decisions
    LOW = 4      # Statistics, monitoring events


@dataclass
class Event:
    """Base event class for the discrete event simulation."""
    timestamp: float
    event_type: EventType
    target: str  # Target node/component identifier
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.MEDIUM
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __lt__(self, other):
        """Compare events for priority queue ordering."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority.value < other.priority.value

    def __repr__(self):
        return (f"Event(type={self.event_type.value}, timestamp={self.timestamp:.3f}, "
                f"target={self.target}, priority={self.priority.name})")


class EventLoop:
    """Central event loop for discrete event simulation."""

    def __init__(self):
        self.event_queue: list[Event] = []
        self.current_time = 0.0
        self.running = False
        self.event_handlers: Dict[EventType, callable] = {}
        self.statistics = {
            'events_processed': 0,
            'total_simulation_time': 0.0,
            'events_by_type': {}
        }

    def schedule_event(self, event: Event):
        """Schedule an event to be processed at a future time."""
        if event.timestamp < self.current_time:
            raise ValueError(f"Cannot schedule event in the past: {event.timestamp} < {self.current_time}")
        heapq.heappush(self.event_queue, event)

    def register_handler(self, event_type: EventType, handler: callable):
        """Register a handler function for a specific event type."""
        self.event_handlers[event_type] = handler

    def step(self) -> bool:
        """Process one event from the queue."""
        if not self.event_queue:
            return False

        event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp

        # Update statistics
        self.statistics['events_processed'] += 1
        event_type_name = event.event_type.value
        self.statistics['events_by_type'][event_type_name] = \
            self.statistics['events_by_type'].get(event_type_name, 0) + 1

        # Process event
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            print(f"Warning: No handler registered for event type {event.event_type.value}")

        return True

    def run(self, max_time: float = float('inf'), max_events: int = None):
        """Run the simulation until no more events or limits reached."""
        self.running = True
        events_processed = 0

        while self.running and self.event_queue:
            if self.current_time > max_time:
                break
            if max_events and events_processed >= max_events:
                break

            if not self.step():
                break
            events_processed += 1

        self.running = False
        self.statistics['total_simulation_time'] = self.current_time

    def stop(self):
        """Stop the simulation loop."""
        self.running = False

    def get_next_event_time(self) -> Optional[float]:
        """Get the timestamp of the next scheduled event."""
        if not self.event_queue:
            return None
        return self.event_queue[0].timestamp

    def peek_next_events(self, count: int = 1) -> list[Event]:
        """Get the next events without removing them from the queue."""
        return self.event_queue[:count]

    def clear_future_events(self, target: str = None) -> int:
        """Clear all future events, optionally filtered by target."""
        if target is None:
            count = len(self.event_queue)
            self.event_queue.clear()
            return count

        original_length = len(self.event_queue)
        self.event_queue = [e for e in self.event_queue if e.target != target]
        return original_length - len(self.event_queue)

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            **self.statistics,
            'current_time': self.current_time,
            'pending_events': len(self.event_queue),
            'running': self.running
        }

    def reset(self):
        """Reset the event loop to initial state."""
        self.event_queue.clear()
        self.current_time = 0.0
        self.running = False
        self.statistics = {
            'events_processed': 0,
            'total_simulation_time': 0.0,
            'events_by_type': {}
        }
