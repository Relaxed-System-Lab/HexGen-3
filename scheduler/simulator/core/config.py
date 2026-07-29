from dataclasses import dataclass
from typing import Optional, Union

@dataclass(frozen=True)
class ParallelConfig:
    """Describe model parallel settings for an engine."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    def with_defaults(self) -> "ParallelConfig":
        """Return a normalized copy with minimum values enforced."""
        return ParallelConfig(
            tensor_parallel_size=max(1, self.tensor_parallel_size),
            pipeline_parallel_size=max(1, self.pipeline_parallel_size),
        )

def _normalize_parallel_config(
    config: Optional[Union["ParallelConfig", int, dict]]
) -> ParallelConfig:
    """Coerce user-provided config values into a ParallelConfig instance."""
    if config is None:
        return ParallelConfig()
    if isinstance(config, ParallelConfig):
        return config.with_defaults()
    if isinstance(config, int):
        return ParallelConfig(tensor_parallel_size=max(1, config))
    if isinstance(config, dict):
        tp = max(1, int(config.get("tensor_parallel_size", 1)))
        pp = max(1, int(config.get("pipeline_parallel_size", 1)))
        return ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)
    raise TypeError(
        f"Unsupported parallel_config type: {type(config)!r}. "
        "Expected ParallelConfig, int, dict, or None."
    )

from typing import List, Tuple
from simulator.core.arrival import ArrivalProcess

@dataclass
class WorkloadConfig:
    """Configuration for a specific model's workload."""
    model_id: str
    arrival_process: ArrivalProcess
    duration: float
    input_dist: Tuple[str, List[float]]  # (type, params)
    output_dist: Tuple[str, List[float]] # (type, params)
    tensor_parallel_size: Optional[int] = None
