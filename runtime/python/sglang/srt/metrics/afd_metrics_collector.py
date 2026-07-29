# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AFD Metrics Collector for autoscaling evaluation.

This module collects real-time metrics for AFD (Attention-FFN Disaggregation)
autoscaling evaluation, including:
- Throughput: Prefill TPS, FFN TPS, Attention TPS, KV-ops/sec
- Hardware utilization: GPU utilization
- Queue metrics: Average queue wait time
- Workload: KV cache uncached token count
"""

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AFDMetricsSample:
    """A single metrics sample."""
    timestamp: float
    worker_type: str  # "prefill", "ffn", "attn", or "unified"
    
    # Workload: KV cache uncached tokens
    workload_uncached_tokens: float  # Rate: uncached tokens per second
    window_s: float = 0.0
    window_received_requests: int = 0
    window_finished_requests: int = 0
    window_failed_requests: int = 0
    window_finished_output_tokens: int = 0
    arrival_rate_rps: float = 0.0
    finished_requests_per_sec: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    running_requests: int = 0
    running_output_tokens: int = 0
    avg_running_output_tokens: float = 0.0
    waiting_requests: int = 0
    
    # Throughput (tokens per second in current window)
    prefill_tps: float = 0.0
    ffn_tps: float = 0.0
    attn_tps: float = 0.0
    attn_kv_ops_per_sec: float = 0.0
    
    # Hardware utilization
    gpu_utilization: float = 0.0
    
    # Queue metrics
    avg_queue_wait_time: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary, ensuring all values are JSON-serializable."""
        def to_python_type(value):
            """Convert value to Python native type (int or float)."""
            if value is None:
                return 0
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    return value.item()
            except (ImportError, AttributeError):
                pass
            if isinstance(value, (str, bool, int, float)):
                return value
            try:
                return float(value)
            except (ValueError, TypeError):
                return str(value)  # Fallback to string representation
        
        result = asdict(self)
        # Convert any Tensor values to Python native types
        for key, value in result.items():
            result[key] = to_python_type(value)
        return result


class GPUUtilizationMonitor:
    """Monitor GPU utilization using nvidia-smi."""
    
    @staticmethod
    def get_current_utilization() -> float:
        """Get current average GPU utilization across all GPUs."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                utilizations = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            utilizations.append(float(line.strip()))
                        except ValueError:
                            pass
                
                if utilizations:
                    return sum(utilizations) / len(utilizations)
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization: {e}")
        
        return 0.0


class AFDMetricsCollector:
    """Collect AFD metrics periodically and write to file."""
    
    def __init__(
        self,
        scheduler,
        output_file: str,
        interval: float = 1.0,
        worker_type: Optional[str] = None,
    ):
        """
        Initialize the AFD metrics collector.
        
        Args:
            scheduler: Scheduler instance to collect metrics from
            output_file: Path to output JSONL file
            interval: Collection interval in seconds
        """
        self.scheduler = scheduler
        self.output_file = Path(output_file)
        self.interval = interval
        self.running = False
        self.collector_thread: Optional[threading.Thread] = None
        
        # Previous values for rate calculation
        self.prev_timestamp: Optional[float] = None
        self.prev_prefill_tokens: int = 0
        self.prev_ffn_tokens: int = 0
        self.prev_attn_tokens: int = 0
        self.prev_attn_kv_ops: int = 0
        self.prev_uncached_tokens: int = 0
        self.prev_received_requests: int = 0
        self.prev_received_input_tokens: int = 0
        self.prev_finished_requests: int = 0
        self.prev_finished_output_tokens: int = 0
        self.prev_failed_requests: int = 0
        
        self.worker_type = worker_type or self._infer_worker_type()
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AFD metrics collector initialized: worker_type={self.worker_type}, output={self.output_file}, interval={self.interval}s")

    def _infer_worker_type(self) -> str:
        from sglang.srt.layers.afd import afd_is_attn, afd_is_ffn, get_afd_perspective

        if get_afd_perspective() is None:
            return "unified"
        if afd_is_attn():
            return "attn"
        if afd_is_ffn():
            return "ffn"
        return "unknown"
    
    def get_metrics_snapshot(self) -> AFDMetricsSample:
        """Collect a snapshot of current metrics."""
        current_time = time.time()
        
        # Helper function to convert to Python native types
        def to_python_type(value):
            """Convert value to Python native type (int or float)."""
            if value is None:
                return 0
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    return value.item()
            except (ImportError, AttributeError):
                pass
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0
        
        # Initialize previous values on first call (do this before any rate calculations)
        if self.prev_timestamp is None:
            self.prev_timestamp = current_time
            self.prev_prefill_tokens = to_python_type(getattr(self.scheduler, 'prefill_tokens_total', 0))
            self.prev_ffn_tokens = to_python_type(getattr(self.scheduler, 'ffn_tokens_total', 0))
            self.prev_attn_tokens = to_python_type(getattr(self.scheduler, 'attn_tokens_total', 0))
            self.prev_attn_kv_ops = to_python_type(getattr(self.scheduler, 'attn_kv_ops_total', 0))
            self.prev_uncached_tokens = to_python_type(getattr(self.scheduler, 'received_uncached_tokens_total', 0))
            self.prev_received_requests = to_python_type(getattr(self.scheduler, 'received_requests_total', 0))
            self.prev_received_input_tokens = to_python_type(getattr(self.scheduler, 'received_input_tokens_total', 0))
            self.prev_finished_requests = to_python_type(getattr(self.scheduler, 'finished_requests_total', 0))
            self.prev_finished_output_tokens = to_python_type(getattr(self.scheduler, 'finished_output_tokens_total', 0))
            self.prev_failed_requests = to_python_type(getattr(self.scheduler, 'failed_requests_total', 0))
            running_requests, running_output_tokens, avg_running_output_tokens = (
                self._get_running_output_shape()
            )
            # Return zeros for first sample
            return AFDMetricsSample(
                timestamp=current_time,
                worker_type=self.worker_type,
                workload_uncached_tokens=0.0,
                running_requests=running_requests,
                running_output_tokens=running_output_tokens,
                avg_running_output_tokens=avg_running_output_tokens,
                waiting_requests=self._get_waiting_requests(),
                prefill_tps=0.0,
                ffn_tps=0.0,
                attn_tps=0.0,
                attn_kv_ops_per_sec=0.0,
                gpu_utilization=GPUUtilizationMonitor.get_current_utilization(),
                avg_queue_wait_time=self._get_avg_queue_wait_time(),
            )
        
        # Get workload: uncached tokens rate (per second) from received requests
        workload_uncached_tokens_rate = self._get_uncached_tokens_rate(current_time)
        (
            window_s,
            window_received_requests,
            window_finished_requests,
            window_failed_requests,
            window_finished_output_tokens,
            arrival_rate_rps,
            finished_requests_per_sec,
            avg_input_tokens,
            avg_output_tokens,
        ) = self._get_request_workload_rates(current_time)
        running_requests, running_output_tokens, avg_running_output_tokens = (
            self._get_running_output_shape()
        )
        
        # Get throughput (calculate rate from counters)
        prefill_tps, ffn_tps, attn_tps, attn_kv_ops_per_sec = self._calculate_throughput(current_time)
        
        # Update timestamp after all calculations
        self.prev_timestamp = current_time
        
        # Get GPU utilization
        gpu_utilization = GPUUtilizationMonitor.get_current_utilization()
        
        # Get average queue wait time
        avg_queue_wait_time = self._get_avg_queue_wait_time()


        return AFDMetricsSample(
            timestamp=current_time,
            worker_type=self.worker_type,
            workload_uncached_tokens=workload_uncached_tokens_rate,
            window_s=window_s,
            window_received_requests=window_received_requests,
            window_finished_requests=window_finished_requests,
            window_failed_requests=window_failed_requests,
            window_finished_output_tokens=window_finished_output_tokens,
            arrival_rate_rps=arrival_rate_rps,
            finished_requests_per_sec=finished_requests_per_sec,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            running_requests=running_requests,
            running_output_tokens=running_output_tokens,
            avg_running_output_tokens=avg_running_output_tokens,
            waiting_requests=self._get_waiting_requests(),
            prefill_tps=prefill_tps,
            ffn_tps=ffn_tps,
            attn_tps=attn_tps,
            attn_kv_ops_per_sec=attn_kv_ops_per_sec,
            gpu_utilization=gpu_utilization,
            avg_queue_wait_time=avg_queue_wait_time,
        )
    
    def _get_uncached_tokens_rate(self, current_time: float) -> float:
        """Get uncached tokens rate (per second) from received requests (workload)."""
        try:
            def to_python_type(value):
                """Convert value to Python native type (int or float)."""
                if value is None:
                    return 0
                try:
                    import torch
                    if isinstance(value, torch.Tensor):
                        return value.item()
                except (ImportError, AttributeError):
                    pass
                try:
                    return int(value)
                except (ValueError, TypeError):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0
            
            current_uncached_tokens = to_python_type(getattr(self.scheduler, 'received_uncached_tokens_total', 0))
            
            time_elapsed = current_time - self.prev_timestamp
            if time_elapsed <= 0:
                return 0.0
            
            uncached_tokens_diff = current_uncached_tokens - self.prev_uncached_tokens

            rate = uncached_tokens_diff / time_elapsed
            
            # Update previous value (timestamp will be updated in get_metrics_snapshot)
            self.prev_uncached_tokens = current_uncached_tokens
            
            return max(0.0, rate)
        except Exception as e:
            logger.debug(f"Failed to get uncached tokens rate: {e}")
            return 0.0

    def _get_request_workload_rates(self, current_time: float) -> tuple:
        """Get request arrival rate and average token shape for the current window."""
        try:
            def to_python_type(value):
                if value is None:
                    return 0
                try:
                    import torch
                    if isinstance(value, torch.Tensor):
                        return value.item()
                except (ImportError, AttributeError):
                    pass
                try:
                    return int(value)
                except (ValueError, TypeError):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0

            time_elapsed = current_time - self.prev_timestamp
            if time_elapsed <= 0:
                return 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

            current_requests = to_python_type(getattr(self.scheduler, 'received_requests_total', 0))
            current_input_tokens = to_python_type(getattr(self.scheduler, 'received_input_tokens_total', 0))
            current_finished_requests = to_python_type(getattr(self.scheduler, 'finished_requests_total', 0))
            current_finished_output_tokens = to_python_type(getattr(self.scheduler, 'finished_output_tokens_total', 0))
            current_failed_requests = to_python_type(getattr(self.scheduler, 'failed_requests_total', 0))

            request_diff = max(0, current_requests - self.prev_received_requests)
            input_diff = max(0, current_input_tokens - self.prev_received_input_tokens)
            finished_request_diff = max(0, current_finished_requests - self.prev_finished_requests)
            finished_output_diff = max(0, current_finished_output_tokens - self.prev_finished_output_tokens)
            failed_request_diff = max(0, current_failed_requests - self.prev_failed_requests)

            self.prev_received_requests = current_requests
            self.prev_received_input_tokens = current_input_tokens
            self.prev_finished_requests = current_finished_requests
            self.prev_finished_output_tokens = current_finished_output_tokens
            self.prev_failed_requests = current_failed_requests

            arrival_rate_rps = request_diff / time_elapsed if request_diff > 0 else 0.0
            finished_requests_per_sec = (
                finished_request_diff / time_elapsed
                if finished_request_diff > 0
                else 0.0
            )
            avg_input_tokens = (
                input_diff / request_diff
                if request_diff > 0
                else 0.0
            )
            avg_output_tokens = (
                finished_output_diff / finished_request_diff
                if finished_request_diff > 0
                else 0.0
            )

            return (
                time_elapsed,
                request_diff,
                finished_request_diff,
                failed_request_diff,
                finished_output_diff,
                arrival_rate_rps,
                finished_requests_per_sec,
                avg_input_tokens,
                avg_output_tokens,
            )
        except Exception as e:
            logger.debug(f"Failed to get request workload rates: {e}")
            return 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    
    def _calculate_throughput(self, current_time: float) -> tuple:
        """Calculate throughput rates from counters."""
        prefill_tps = 0.0
        ffn_tps = 0.0
        attn_tps = 0.0
        attn_kv_ops_per_sec = 0.0
        
        time_elapsed = current_time - self.prev_timestamp
        if time_elapsed <= 0:
            logger.info("Time elapsed is zero, returning zero throughput")
            return prefill_tps, ffn_tps, attn_tps, attn_kv_ops_per_sec
        
        # Get current counter values (convert to Python native types if they are Tensors)
        def to_python_type(value):
            """Convert value to Python native type (int or float)."""
            if value is None:
                return 0
            # Check if it's a PyTorch tensor
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    return value.item()  # Convert tensor to Python scalar
            except (ImportError, AttributeError):
                pass
            # Try to convert to int first, then float
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0
        
        current_prefill_tokens = to_python_type(getattr(self.scheduler, 'prefill_tokens_total', 0))
        current_ffn_tokens = to_python_type(getattr(self.scheduler, 'ffn_tokens_total', 0))
        current_attn_tokens = to_python_type(getattr(self.scheduler, 'attn_tokens_total', 0))
        current_attn_kv_ops = to_python_type(getattr(self.scheduler, 'attn_kv_ops_total', 0))

        # diff
        prefill_tokens_diff = current_prefill_tokens - self.prev_prefill_tokens
        ffn_tokens_diff = current_ffn_tokens - self.prev_ffn_tokens
        attn_tokens_diff = current_attn_tokens - self.prev_attn_tokens
        attn_kv_ops_diff = current_attn_kv_ops - self.prev_attn_kv_ops

        # Calculate rates
        prefill_tps = prefill_tokens_diff / time_elapsed
        ffn_tps = ffn_tokens_diff / time_elapsed
        attn_tps = attn_tokens_diff / time_elapsed
        attn_kv_ops_per_sec = attn_kv_ops_diff / time_elapsed

        
        # Update previous values (timestamp will be updated in get_metrics_snapshot)
        self.prev_prefill_tokens = current_prefill_tokens
        self.prev_ffn_tokens = current_ffn_tokens
        self.prev_attn_tokens = current_attn_tokens
        self.prev_attn_kv_ops = current_attn_kv_ops
        
        return prefill_tps, ffn_tps, attn_tps, attn_kv_ops_per_sec
    
    def _get_avg_queue_wait_time(self) -> float:
        """Get average queue wait time for requests currently waiting."""
        try:
            waiting_queue = getattr(self.scheduler, 'waiting_queue', [])
            if not waiting_queue:
                return 0.0
            
            # Use time.perf_counter() to match queue_time_start which is also set using time.perf_counter()
            current_time = time.perf_counter()
            total_wait_time = 0.0
            count = 0
            
            for req in waiting_queue:
                if hasattr(req, 'queue_time_start') and req.queue_time_start is not None:
                    wait_time = current_time - req.queue_time_start
                    total_wait_time += wait_time
                    count += 1
            
            if count > 0:
                return total_wait_time / count
            return 0.0
        except Exception as e:
            logger.debug(f"Failed to get queue wait time: {e}")
            return 0.0

    def _get_running_requests(self) -> int:
        return self._get_running_output_shape()[0]

    def _get_running_output_shape(self) -> tuple:
        try:
            running_batch = getattr(self.scheduler, 'running_batch', None)
            reqs = getattr(running_batch, 'reqs', []) if running_batch is not None else []
            active_reqs = []
            for req in reqs:
                if str(getattr(req, 'rid', '')).startswith('HEALTH_CHECK'):
                    continue
                if callable(getattr(req, 'finished', None)) and req.finished():
                    continue
                active_reqs.append(req)
            output_tokens = sum(
                len(getattr(req, 'output_ids', []) or []) for req in active_reqs
            )
            request_count = len(active_reqs)
            avg_output_tokens = (
                output_tokens / request_count if request_count > 0 else 0.0
            )
            return request_count, output_tokens, avg_output_tokens
        except Exception:
            return 0, 0, 0.0

    def _get_waiting_requests(self) -> int:
        try:
            waiting = len(getattr(self.scheduler, 'waiting_queue', []) or [])
            waiting += len(getattr(self.scheduler, 'grammar_queue', []) or [])

            for name in (
                'disagg_decode_prealloc_queue',
                'disagg_decode_transfer_queue',
                'disagg_prefill_bootstrap_queue',
            ):
                queue = getattr(self.scheduler, name, None)
                waiting += len(getattr(queue, 'queue', []) or [])

            prealloc_queue = getattr(
                self.scheduler,
                'disagg_decode_prealloc_queue',
                None,
            )
            waiting += len(getattr(prealloc_queue, 'retracted_queue', []) or [])
            waiting += len(
                getattr(self.scheduler, 'disagg_prefill_inflight_queue', []) or []
            )
            return waiting
        except Exception:
            return 0
    
    def _write_sample(self, sample: AFDMetricsSample):
        """Write a sample to the output file."""
        try:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(sample.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write metrics sample: {e}")
    
    def _collect_loop(self):
        """Main collection loop (runs in separate thread)."""
        logger.info(f"AFD metrics collector started (worker_type={self.worker_type})")
        
        while self.running:
            try:
                sample = self.get_metrics_snapshot()
                self._write_sample(sample)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            # Sleep until next interval
            time.sleep(self.interval)
        
        logger.info("AFD metrics collector stopped")
    
    def start(self):
        """Start the metrics collector."""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.collector_thread.start()
        logger.info("AFD metrics collector started")
    
    def stop(self):
        """Stop the metrics collector."""
        if not self.running:
            return
        
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        logger.info("AFD metrics collector stopped")
