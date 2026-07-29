from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
import json
import hashlib


@dataclass
class TraceEvent:
    name: str
    cat: str
    ph: str
    pid: Union[int, str]
    tid: Union[int, str]

    ts: int  # in microseconds
    args: Optional[dict] = None
    dur: Optional[int] = None
    cname: Optional[str] = None  # Color name for Chrome tracing

    def to_dict(self) -> Dict[str, Any]:
        """Convert TraceEvent to dictionary format for Chrome tracing."""
        result = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts
        }

        if self.args is not None:
            result["args"] = self.args

        if self.dur is not None:
            result["dur"] = self.dur

        if self.cname is not None:
            result["cname"] = self.cname

        return result


def get_request_color(request_id: str) -> str:
    """
    Generate a consistent color for a request ID based on its hash.

    Uses Chrome tracing's built-in color names that are guaranteed to work.
    These are the standard color names that Chrome tracing recognizes.

    Args:
        request_id: The request ID to generate a color for

    Returns:
        A color name string for Chrome tracing
    """
    # Use a focused set of visually distinct Chrome tracing colors
    colors = [
        "generic_work",      # Default blue/gray work color
        "good",              # Greenish color
        "rail_response",     # Blue/purple for response operations
        "rail_animation",    # Orange for animation-like operations
        "rail_load",         # Yellow for loading operations
        "startup",           # Red/pink for startup-like operations
        "cq_build_running",  # Another distinct color
        "cq_build_passed",   # Another distinct color
        "thread_state_running",  # Green for running state
        "thread_state_iowait",   # Orange for I/O wait
    ]

    # Generate a consistent hash of the request ID
    hash_obj = hashlib.md5(request_id.encode())
    hash_int = int(hash_obj.hexdigest(), 16)

    # Map hash to a color
    color_index = hash_int % len(colors)
    return colors[color_index]


def export_chrome_trace(events: List[TraceEvent], output_file: str) -> None:
    """
    Export trace events to Chrome tracing JSON format.

    This format can be loaded into chrome://tracing for visualization.

    Args:
        events: List of TraceEvent objects to export
        output_file: Path to output JSON file
    """
    # Collect unique process IDs to add process metadata
    process_ids = set()
    for event in events:
        process_ids.add(event.pid)

    # Create process metadata for better visualization
    process_metadata = {}
    for pid in process_ids:
        if pid == "request_tracker":
            process_metadata[pid] = {
                "name": "Request Lifecycle Tracker",
                "sort_index": -1  # Show at the top
            }
        elif pid == "cluster":
            process_metadata[pid] = {
                "name": "Cluster Manager",
                "sort_index": 0
            }
        elif isinstance(pid, str) and pid.startswith("node_"):
            process_metadata[pid] = {
                "name": f"Engine: {pid}",
                "sort_index": int(pid.split("_")[1]) if "_" in pid else 1
            }
        else:
            process_metadata[pid] = {
                "name": f"Process: {pid}",
                "sort_index": 2
            }

    # Convert process metadata to Chrome trace format
    process_events = []
    for pid, metadata in process_metadata.items():
        process_events.append({
            "name": "process_name",
            "ph": "M",  # Metadata event
            "pid": pid,
            "args": {"name": metadata["name"]}
        })
        process_events.append({
            "name": "process_sort_index",
            "ph": "M",  # Metadata event
            "pid": pid,
            "args": {"sort_index": metadata["sort_index"]}
        })

    trace_data = {
        "traceEvents": process_events + [event.to_dict() for event in events],
        "displayTimeUnit": "ms"
    }

    with open(output_file, 'w') as f:
        json.dump(trace_data, f, indent=2)


def export_chrome_trace_from_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Export trace events from simulation results to Chrome tracing format.

    Args:
        results: Results dictionary from cluster simulation
        output_file: Path to output JSON file
    """
    if "trace_events" not in results:
        raise ValueError("No trace_events found in results dictionary")

    events = results["trace_events"]
    export_chrome_trace(events, output_file)