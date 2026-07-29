# AF disaggregation

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
# =========

import itertools
from collections import deque
from enum import Enum, auto
from typing import Any, Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from functools import cache

import sys
import os
import time
import logging
import signal
import multiprocessing
import atexit

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    # Standard colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

# Color mapping for TP ranks (for easier debugging)
TP_COLORS = [
    Colors.BRIGHT_CYAN,      # TP0 - Cyan
    Colors.BRIGHT_YELLOW,    # TP1 - Yellow
    Colors.BRIGHT_GREEN,     # TP2 - Green
    Colors.BRIGHT_MAGENTA,   # TP3 - Magenta
    Colors.BRIGHT_BLUE,      # TP4 - Blue
    Colors.BRIGHT_RED,       # TP5 - Red
]

def get_tp_color_prefix(tp_rank: int) -> str:
    """Get color-coded prefix for TP rank logging."""
    if tp_rank < len(TP_COLORS):
        color = TP_COLORS[tp_rank]
        return f"{color}TP{tp_rank}{Colors.RESET}"
    else:
        # Fallback for ranks beyond our color list
        return f"TP{tp_rank}"

def log_with_tp_color(tp_rank: int, message: str, level: int = logging.INFO):
    """Log message with TP rank color coding."""
    prefix = get_tp_color_prefix(tp_rank)
    colored_message = f"[{prefix}] {message}"
    logger.log(level, colored_message)

def format_afd_debug(tp_rank: int, message: str) -> str:
    """Format AFD DEBUG message with colored TP rank prefix."""
    tp_prefix = get_tp_color_prefix(tp_rank)
    return f"[AFD DEBUG] [{tp_prefix}] {message}"

import torch
from torch import nn
import torch.distributed as dist
import zmq

from sglang.srt.nvtx_utils import nvtx_range
from sglang.srt.managers.schedule_batch import global_server_args_dict, get_global_dp_rank
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.afd_type import AFDPerspective

from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    ScatterMode,
)
from sglang.srt.distributed import (
    get_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_attention_tp_group,
    attn_tp_all_gather_into_tensor,
    dp_gather_partial,
    dp_scatter,
)

class AFDForwardStage(Enum):
    AFD_FORWARD_STAGE_A = auto()
    AFD_FORWARD_STAGE_F = auto()

class AFDStageScheduleGenerator:
    Schedule = List[Tuple[AFDForwardStage, int, int]]
    @staticmethod
    def ffn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        for l, m in itertools.product(range(num_layers), range(m_stage)):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l, m))
        return schedule
    @staticmethod
    def attn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        if num_layers == 1:
            return (
                [(AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)]
                +
                [(AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)]
            )
        for l, m in itertools.product(range(num_layers + 1), range(m_stage)):
            if l > 0:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l - 1, m))
            if l < num_layers:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
        return schedule

class FifoTensorCommunicator(ABC):
    @abstractmethod
    def recv_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, x: torch.Tensor):
        raise NotImplementedError

class ZMQSimpleTensorCommunicator(FifoTensorCommunicator):
    def __init__(self, afd_perspective: AFDPerspective):
        super().__init__()
        self.zmq_context = zmq.Context()

        self.start_lport = (
            self.get_ffn_port() if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self.get_attn_port()
        )

        self.start_dport = (
            self.get_attn_port() if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self.get_ffn_port()
        )

    def get_ffn_port(self) -> int:
        return 40000

    def get_attn_port(self) -> int:
        return 50000

    def get_lport(self) -> int:
        return self.start_lport + 1 + dist.get_rank()

    def get_dport(self) -> int:
        return self.start_dport + 1 + dist.get_rank()

    def get_current_cuda_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot get current CUDA device.")
        device_index = torch.cuda.current_device()
        return torch.device(f"cuda:{device_index}")

    @cache
    def get_push_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PUSH)
        socket.connect(f"tcp://localhost:{self.get_dport()}")
        return socket

    @cache
    def get_pull_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.get_lport()}")
        return socket

    def recv_tensor(self) -> torch.Tensor:
        with nvtx_range("afd.socket.recv.get_socket"):
            socket = self.get_pull_socket()
        with nvtx_range("afd.socket.recv.recv_pyobj"):
            x = socket.recv_pyobj()
        assert isinstance(x, torch.Tensor)
        size_info = f"[numel={x.numel()},bytes={x.numel() * x.element_size()}]"
        with nvtx_range(f"afd.socket.recv.to_cuda{size_info}"):
            return x.to(self.get_current_cuda_device())

    def send_tensor(self, x: torch.Tensor):
        with nvtx_range("afd.socket.send.get_socket"):
            socket = self.get_push_socket()
        size_info = f"[numel={x.numel()},bytes={x.numel() * x.element_size()}]"
        with nvtx_range(f"afd.socket.send.send_pyobj{size_info}"):
            socket.send_pyobj(x)

class StepMeshTensorCache(object):
    def __init__(self):
        self.push_tensor = None
        self.pull_tensor = None

        self.push_key = 0
        self.pull_key = 0

        self.h = None

def stepmesh_scheduler():
    os.environ['DMLC_ROLE'] = 'scheduler'

    port = os.environ.get('DMLC_PS_ROOT_PORT', 'unknown')
    host = os.environ.get('DMLC_NODE_HOST', 'unknown')
    stepmesh_gpu = os.environ.get('STEPMESH_GPU', 'not_set')
    rank_offset = os.environ.get('DMLC_RANK_OFFSET', 'not_set')
    logger.info(
        f"StepMesh scheduler starting: port={port}, host={host}, "
        f"STEPMESH_GPU={stepmesh_gpu}, DMLC_RANK_OFFSET={rank_offset}, "
        f"DMLC_NUM_WORKER={os.environ.get('DMLC_NUM_WORKER', 'not_set')}, "
        f"DMLC_NUM_SERVER={os.environ.get('DMLC_NUM_SERVER', 'not_set')}"
    )
    import fserver_lib as f

    f.init()
    logger.info(f"StepMesh scheduler init done: port={port}")

    # Flag to indicate shutdown request
    should_stop = False

    def signal_handler(signum, frame):
        nonlocal should_stop
        logger.info(f"StepMesh scheduler received signal {signum}, shutting down...")
        should_stop = True

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Main loop: check should_stop flag instead of infinite sleep
    try:
        while not should_stop:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("StepMesh scheduler received KeyboardInterrupt, shutting down...")
        should_stop = True

    # Cleanup RDMA resources before exiting
    logger.info("StepMesh scheduler cleaning up RDMA resources...")
    try:
        f.stop()
        logger.info("StepMesh scheduler stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping StepMesh scheduler: {e}")
    logger.info(f"StepMesh scheduler exited: port={port}")

class StepMeshTensorCommunicator(FifoTensorCommunicator):
    def __init__(self, afd_perspective: AFDPerspective):
        self.perspective = afd_perspective

        super().__init__()

        # only TP0 process initializes StepMesh connection
        tp_rank = get_tensor_model_parallel_rank()
        
        if tp_rank == 0:
            import fserver_lib as f

            # Initialize scheduler_process to None
            self.scheduler_process = None

            self.start_stepmesh_scheduler()

            time.sleep(10) # wait scheduler

            role = os.environ.get('DMLC_ROLE', 'unknown')
            dp_rank = self.get_dp_rank()
            current_device = torch.cuda.current_device()
            stepmesh_gpu = os.environ.get('STEPMESH_GPU', 'not_set')
            stepmesh_port = 8123 + dp_rank
            logger.info(
                f"[Sglang] "
                f"StepMesh {role} init: dp_rank={dp_rank}, tp_rank=0, "
                f"current_device=cuda:{current_device}, STEPMESH_GPU={stepmesh_gpu}, "
                f"port={stepmesh_port}, DMLC_RANK_OFFSET={os.environ.get('DMLC_RANK_OFFSET', 'not_set')}"
                f"PS_VERBOSE={os.environ.get('PS_VERBOSE', 'not_set')}"
            )
            f.init()
            logger.info(
                f"[Sglang] "
                f"StepMesh {role} init done: dp_rank={dp_rank}, tp_rank=0, current_device=cuda:{current_device}")

            self.worker_num = int(os.environ['DMLC_NUM_WORKER'])

            self.key = 0
            self.gpu = torch.cuda.current_device()
            self.tensor_shape = None

            self.f = f

            self.comm_ids = []
            self.waits = []
            self.free_tensors = {}
            self.register_buf = {}
        else:
            logger.info(f"StepMesh: tp_rank={tp_rank}, skipping StepMesh initialization (only TP0 establishes connection)")
            self.f = None
            self.worker_num = None
            self.key = None
            self.gpu = None
            self.tensor_shape = None
            self.comm_ids = None
            self.waits = None
            self.free_tensors = None
            self.register_buf = None
            self.scheduler_process = None

    def env_def(self, env, v):
        if os.environ.get(env) == None:
            os.environ[env] = v

    def get_node_ip(self):
        if os.environ.get("DMLC_NODE_HOST") != None:
            return

        import psutil

        interface_name = os.environ.get("MLC_INTERFACE")

        interfaces = psutil.net_if_addrs()

        if interface_name not in interfaces:
            print("Invalid MLC_INTERFACE %s" % interface_name)
            return

        for addr in interfaces[interface_name]:
            if addr.family == 2:  # socket.AF_INET
                os.environ["DMLC_NODE_HOST"] = addr.address
                logger.info(f"StepMesh: DMLC_NODE_HOST={os.environ.get('DMLC_NODE_HOST', 'not_set')}")
                break

    def get_dp_rank(self):
        """Get dp_rank from global state or fallback methods."""
        # First, try to get from global state (set by Scheduler)
        dp_rank = get_global_dp_rank()
        if dp_rank is not None:
            return dp_rank
        
        # Default to 0 if not found
        # logger.warning(f"StepMesh: Could not determine dp_rank, defaulting to 0")
        return 0

    def start_stepmesh_scheduler(self):
        tp_rank = get_tensor_model_parallel_rank()
        if tp_rank != 0:
            return
        
        self.get_node_ip()

        # Get dp_rank to determine which scheduler instance to use
        dp_rank = self.get_dp_rank()
        
        # Calculate port based on dp_rank: base_port + dp_rank
        # This ensures each dp_rank uses a different scheduler instance
        # For example: dp_rank=0 uses port 8123, dp_rank=1 uses port 8124
        base_port = 8123
        stepmesh_port = base_port + dp_rank
        logger.info(f"StepMesh: dp_rank={dp_rank}, tp_rank=0, using port={stepmesh_port}")

        # StepMesh rank calculation: rank = group_size * node_rank + gpu + offset
        # With DMLC_GROUP_SIZE=1, DMLC_NODE_RANK=0: rank = 0 + gpu + offset = gpu + offset
        # We want rank = 0 for each dp_rank group, so offset = -gpu
        # 
        # IMPORTANT: STEPMESH_GPU should use the LOGICAL device ID (from torch.cuda.current_device())
        # This is the device ID that PyTorch sees, which may differ from physical GPU ID
        # if CUDA_VISIBLE_DEVICES is set. StepMesh will use this to call cudaSetDevice(),
        # so it must match PyTorch's logical device ID.
        logical_gpu_id = torch.cuda.current_device()
        
        # Get physical GPU ID for debugging (if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(logical_gpu_id)
            physical_gpu_id = pynvml.nvmlDeviceGetPciBusId(handle)
        except:
            physical_gpu_id = "unknown"
        
        gpu = str(logical_gpu_id)
        
        self.env_def('DMLC_NODE_RANK',    '0')
        self.env_def('DMLC_NUM_SERVER',   '1')
        self.env_def('DMLC_NUM_WORKER',   '1')
        self.env_def('DMLC_GROUP_SIZE',   '1')
        self.env_def('DMLC_PS_ROOT_PORT', str(stepmesh_port))
        self.env_def('DMLC_ENABLE_RDMA', 'ibverbs')
        
        # Don't override STEPMESH_GPU if it's already set (e.g., by stepmesh_connector.py)
        # Only set it if it's not already set
        if 'STEPMESH_GPU' not in os.environ:
            self.env_def('STEPMESH_GPU', gpu)
        
        # Key fix: Set DMLC_RANK_OFFSET to ensure rank = 0 for each dp_rank group
        # rank = group_size * node_rank + gpu + offset = 1 * 0 + gpu + offset = gpu + offset
        # We want rank = 0, so offset = -gpu
        # Use the same logical_gpu_id that STEPMESH_GPU uses
        self.env_def('DMLC_RANK_OFFSET',  str(-logical_gpu_id))

        if afd_is_attn():
            os.environ['DMLC_ROLE'] = 'worker'
        else:
            os.environ['DMLC_ROLE'] = 'server'

        if os.environ['DMLC_ROLE'] != 'worker':
            return

        if os.environ.get('DMLC_NODE_RANK') != '0':
            return

        # Since each process has its own environment variables,
        # we don't need to differentiate the flag by dp_rank
        if os.environ.get('STEPMESH_SCHEDULER_STARTED') == '1':
            return

        os.environ["DMLC_PS_ROOT_URI"] = os.environ["DMLC_NODE_HOST"]

        p = multiprocessing.Process(target=stepmesh_scheduler)
        # Set daemon=False to ensure proper cleanup on parent exit
        # The parent process will handle cleanup explicitly
        p.daemon = False
        p.start()
        
        # Save process reference for cleanup
        self.scheduler_process = p
        logger.info(f"StepMesh scheduler process started with PID: {p.pid}")
        
        # Register cleanup function to be called on program exit
        # This ensures the scheduler process is properly terminated even if
        # the parent process exits unexpectedly (e.g., Ctrl+C)
        # Use closure to capture the process reference
        def cleanup_on_exit(proc=p):
            if proc is not None and proc.is_alive():
                logger.info(f"Cleaning up StepMesh scheduler process (PID: {proc.pid}) on exit")
                try:
                    proc.terminate()
                    proc.join(timeout=3)
                    if proc.is_alive():
                        logger.warning("StepMesh scheduler process did not terminate gracefully, killing it")
                        proc.kill()
                        proc.join(timeout=1)
                    logger.info("StepMesh scheduler process cleaned up successfully on exit")
                except Exception as e:
                    logger.error(f"Error cleaning up scheduler process on exit: {e}")
        
        # Register the cleanup function (atexit handlers are called in reverse order of registration)
        atexit.register(cleanup_on_exit)


    def attn_send(self, x):
        if self.f is None:
            raise RuntimeError("StepMesh connection not initialized. This should only be called from TP0.")
        
        with nvtx_range("afd.attn_send.alloc_tensor"):
            free = self.free_tensors.get(x.shape)
            if free == None:
                self.free_tensors[x.shape] = []
                free = self.free_tensors[x.shape]

            if len(free) < 1:
                self.key += 2

                t = StepMeshTensorCache()

                t.push_tensor = torch.empty_like(x)
                t.pull_tensor = torch.empty_like(x)
                t.pull_tensor.zero_()

                t.push_key = self.key
                t.pull_key = self.key + 1
                
            else:
                t = free.pop(0)

        with nvtx_range("afd.attn_send.copy"):
            t.push_tensor.copy_(x)

        with nvtx_range("afd.attn_send.push_pull"):
            h = self.f.push_pull(
                    [t.push_tensor],
                    [t.push_key],
                    [t.pull_tensor],
                    [t.pull_key])

        t.h = h

        with nvtx_range("afd.attn_send.append_waits"):
            self.waits.append(t)

    def attn_recv(self):
        if self.f is None:
            raise RuntimeError("StepMesh connection not initialized. This should only be called from TP0.")
        
        with nvtx_range("afd.attn_recv.pop_waits"):
            t = self.waits.pop(0)
        
        with nvtx_range("afd.attn_recv.wait"):
            self.f.wait(t.h)

        with nvtx_range("afd.attn_recv.device_check"):
            # Debug: Check device mismatch
            current_device = torch.cuda.current_device()
            tensor_device = t.pull_tensor.device.index
            dp_rank = self.get_dp_rank()
            if tensor_device != current_device:
                logger.warning(
                    f"StepMesh attn_recv: Device mismatch detected! "
                    f"dp_rank={dp_rank}, expected cuda:{current_device}, but tensor is on cuda:{tensor_device}. "
                    f"STEPMESH_GPU={os.environ.get('STEPMESH_GPU', 'not_set')}, "
                    f"Moving tensor to correct device."
                )
                t.pull_tensor = t.pull_tensor.to(f"cuda:{current_device}")

        with nvtx_range("afd.attn_recv.append_free"):
            self.free_tensors[t.push_tensor.shape].append(t)
        
        with nvtx_range("afd.attn_recv.clone"):
            result = t.pull_tensor.clone()
        
        return result

    def ffn_send(self, x):
        if self.f is None:
            raise RuntimeError("StepMesh connection not initialized. This should only be called from TP0.")
        
        with nvtx_range("afd.ffn_send.alloc_tensor"):
            free = self.free_tensors.get(x.shape)
            if free == None:
                self.free_tensors[x.shape] = []
                free = self.free_tensors[x.shape]

            if len(free) < 1:
                t = torch.empty_like(x)
            else:
                t = free.pop(0)

        with nvtx_range("afd.ffn_send.copy"):
            t.copy_(x)
        
        with nvtx_range("afd.ffn_send.pop_comm_id"):
            c = self.comm_ids.pop(0)
        
        with nvtx_range("afd.ffn_send.respond"):
            try:
                self.f.respond([t], c, True)
            except Exception as e:
                logger.error(
                    f"[StepMesh] ffn_send: respond() FAILED for handler={c}: {e}"
                )
                import traceback
                logger.error(traceback.format_exc())
                raise

        with nvtx_range("afd.ffn_send.append_free"):
            free.append(t)

    def ffn_recv(self):
        if self.f is None:
            raise RuntimeError("StepMesh connection not initialized. This should only be called from TP0.")
        
        current_device = torch.cuda.current_device()
        dp_rank = self.get_dp_rank()

        # with nvtx_range("afd.ffn_recv.memory_check"):
        #     # Check GPU memory before calling get_batch() to avoid OOM
        #     # StepMesh's get_batch() may trigger RDMA operations that allocate GPU memory
        #     # StepMesh uses cudaMalloc directly, which requires contiguous memory
        #     # PyTorch's memory_reserved may include fragmented memory that cudaMalloc cannot use
        #     current_device = torch.cuda.current_device()
        #     dp_rank = self.get_dp_rank()
            
        #     # Get memory stats for monitoring
        #     if torch.cuda.is_available():
        #         memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**3)  # GB
        #         memory_reserved = torch.cuda.memory_reserved(current_device) / (1024**3)  # GB
        #         memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
        #         memory_free = memory_total - memory_reserved
                
        #         # Log memory usage periodically (every 10 calls) or if memory is low
        #         if not hasattr(self, '_ffn_recv_call_count'):
        #             self._ffn_recv_call_count = 0
        #         self._ffn_recv_call_count += 1
                
        #         if self._ffn_recv_call_count % 10 == 0 or memory_free < 1.0:  # Log every 10 calls or if < 1GB free
        #             logger.info(
        #                 f"StepMesh ffn_recv memory: dp_rank={dp_rank}, "
        #                 f"allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB, "
        #                 f"free={memory_free:.2f}GB, total={memory_total:.2f}GB, "
        #                 f"registered_buffers={len(self.register_buf)}"
        #             )
                
        #         # CRITICAL: StepMesh's cudaMalloc needs contiguous memory
        #         # PyTorch's "free" memory may be fragmented, so we need to be more aggressive
        #         # Clear cache if free memory is less than 5GB to ensure StepMesh can allocate
        #         if memory_free < 5.0:  # Less than 5GB free - be more aggressive
        #             logger.warning(
        #                 f"StepMesh ffn_recv: Low GPU memory detected before get_batch()! "
        #                 f"dp_rank={dp_rank}, free={memory_free:.2f}GB, reserved={memory_reserved:.2f}GB. "
        #                 f"Clearing PyTorch cache to free fragmented memory for StepMesh cudaMalloc."
        #             )
        #             # Synchronize all CUDA operations before clearing cache
        #             torch.cuda.synchronize()
        #             torch.cuda.empty_cache()
        #             torch.cuda.synchronize()  # Ensure cache clear is complete
                    
        #             # Re-check after clearing cache
        #             memory_reserved_after = torch.cuda.memory_reserved(current_device) / (1024**3)
        #             memory_free_after = memory_total - memory_reserved_after
        #             logger.info(
        #                 f"StepMesh ffn_recv: After cache clear, "
        #                 f"reserved={memory_reserved_after:.2f}GB, free={memory_free_after:.2f}GB"
        #             )
                    
        #             # If still low after clearing, try to limit registered buffers
        #             if memory_free_after < 3.0 and len(self.register_buf) > 20:
        #                 logger.warning(
        #                     f"StepMesh ffn_recv: Memory still low after cache clear! "
        #                     f"free={memory_free_after:.2f}GB, registered_buffers={len(self.register_buf)}. "
        #                     f"Consider reducing batch size or registered buffer count."
        #                 )
            
        #     # Synchronize before calling get_batch() to ensure all previous operations complete
        #     # This helps prevent memory fragmentation
        #     torch.cuda.synchronize()
        
        with nvtx_range("afd.ffn_recv.get_batch"):
            batches = self.f.get_batch()

        assert len(batches) == 1, "just handle for one worker"

        with nvtx_range("afd.ffn_recv.extract_batch"):
            x = batches[0][1][0]
            key = batches[0][2][0]
            handler = batches[0][0]
            self.comm_ids.append(handler)

        with nvtx_range("afd.ffn_recv.buffer_management"):
            if self.register_buf.get(key) == None:
                with nvtx_range("afd.ffn_recv.find_reuse_buffer"):
                    # Check if we can reuse an existing buffer with the same shape
                    # This helps reduce memory usage by reusing buffers
                    reused_buffer = False
                    for existing_key, existing_buf in self.register_buf.items():
                        if existing_buf.shape == x.shape and existing_buf.dtype == x.dtype:
                            # Reuse existing buffer by updating the key mapping
                            logger.debug(
                                f"StepMesh ffn_recv: Reusing buffer for key={key} "
                                f"(was key={existing_key}), shape={x.shape}, dtype={x.dtype}"
                            )
                            self.register_buf[key] = existing_buf
                            # Note: We keep the old key mapping too, as StepMesh may still reference it
                            # This is a trade-off between memory and correctness
                            reused_buffer = True
                            break
                
                if not reused_buffer:
                    with nvtx_range("afd.ffn_recv.alloc_buffer"):
                        # Ensure we're on the correct device before creating buffer and calling register_recv_buffer
                        # StepMesh may check the current CUDA device when registering the buffer
                        torch.cuda.set_device(current_device)
                        
                        # Allocate buffer with error handling
                        try:
                            y = torch.empty(x.shape, dtype=x.dtype, device=f"cuda:{current_device}")
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.error(
                                    f"StepMesh ffn_recv: OOM when allocating buffer! "
                                    f"dp_rank={dp_rank}, shape={x.shape}, dtype={x.dtype}, "
                                    f"registered_buffers={len(self.register_buf)}. "
                                    f"Attempting emergency cleanup."
                                )
                            else:
                                raise

                    with nvtx_range("afd.ffn_recv.register_buffer"):
                        # Ensure device is set before calling register_recv_buffer
                        # StepMesh may use the current CUDA device context
                        torch.cuda.set_device(current_device)
                        self.f.register_recv_buffer(y, [0], [key])

                    with nvtx_range("afd.ffn_recv.store_buffer"):
                        self.register_buf[key] = y

        with nvtx_range("afd.ffn_recv.clone"):
            result = x.clone()

        return result

    def recv_tensor(self) -> torch.Tensor:
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            return self.attn_recv()
        else:
            return self.ffn_recv()

    def send_tensor(self, x: torch.Tensor):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            self.attn_send(x)
        else:
            self.ffn_send(x)

    def cleanup(self):
        """Cleanup StepMesh scheduler process and RDMA resources."""
        if hasattr(self, 'scheduler_process') and self.scheduler_process is not None:
            if self.scheduler_process.is_alive():
                logger.info(f"Terminating StepMesh scheduler process (PID: {self.scheduler_process.pid})")
                try:
                    # Send SIGTERM to gracefully shutdown
                    self.scheduler_process.terminate()
                    # Wait up to 5 seconds for graceful shutdown
                    self.scheduler_process.join(timeout=5)
                    if self.scheduler_process.is_alive():
                        logger.warning("StepMesh scheduler process did not terminate gracefully, killing it")
                        self.scheduler_process.kill()
                        self.scheduler_process.join(timeout=2)
                    logger.info("StepMesh scheduler process terminated successfully")
                except Exception as e:
                    logger.error(f"Error terminating StepMesh scheduler process: {e}")
                finally:
                    self.scheduler_process = None
        
        # Cleanup fserver_lib resources if available
        if hasattr(self, 'f') and self.f is not None:
            try:
                self.f.stop()
                logger.info("StepMesh fserver_lib stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping StepMesh fserver_lib: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            # Ignore exceptions in destructor to avoid masking real errors
            pass

@cache
def get_tensor_communicator() -> FifoTensorCommunicator:
    afd_perspective = get_afd_perspective()
    if afd_perspective is not None:
        if os.environ.get("MLC_INTERFACE"):
            return StepMeshTensorCommunicator(afd_perspective)
        else:
            return ZMQSimpleTensorCommunicator(afd_perspective)
    else:
        raise NotImplementedError

def get_afd_mirco_batch() -> int:
    afd_mirco_batch = global_server_args_dict.get("afd_mirco_batch")
    return afd_mirco_batch

def get_afd_perspective() -> Optional[AFDPerspective]:
    p = global_server_args_dict.get("afd_perspective")
    if p is None:
        return None
    if isinstance(p, AFDPerspective):
        return p
    if isinstance(p, str):
        try:
            return AFDPerspective(p)
        except ValueError:
            return None
    return None

def afd_is_ffn():
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_FFN

def afd_is_attn():
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_ATTN

def model_forward_afd_split_inputs(
    layers,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_data_scatter_mode: ScatterMode,
):  
    def _model_forward_afd_split_inputs_raw(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        ) -> List[Dict]:
        return [
            dict(
                **_model_forward_filter_inputs(
                    hidden_states=hidden_states,
                    residual=residual,
                    positions=positions,
                    output_forward_batch=output_forward_batch,
                    afd_subbatch_index=afd_subbatch_index,
                ),
                **({}),
            )
            for afd_subbatch_index, output_forward_batch in enumerate(
                forward_batch.afd_children
            )
        ]

    def _model_forward_filter_inputs(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        output_forward_batch: ForwardBatch,
        afd_subbatch_index: int,
    ) -> Dict:
        token_slice = slice(*output_forward_batch.afd_parent_token_range)
        return dict(
            hidden_states=hidden_states[token_slice],
            residual=None if residual is None else residual[token_slice],
            positions=positions[token_slice],
            forward_batch=output_forward_batch,
            afd_subbatch_index=afd_subbatch_index,
        )

    layer_input_scatter_mode = layers[0].layer_scatter_modes.layer_input_mode
    afd_splitter_scatter_mode = ScatterMode.TP_ATTN_FULL
    context = CommunicateContext.init_new()

    # memory monitoring
    if torch.cuda.is_available():
        mem_before_comm = torch.cuda.memory_allocated() / 1024**3
    
    hidden_states, residual = CommunicateSummableTensorPairFn.execute(
        hidden_states_input_mode=input_data_scatter_mode,
        residual_input_mode=input_data_scatter_mode,
        output_mode=afd_splitter_scatter_mode,
        hidden_states=hidden_states,
        residual=residual,
        forward_batch=forward_batch,
        context=context,
    )

    inputs_arr = _model_forward_afd_split_inputs_raw(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
    )

    def _post_transform(hidden_states, residual, forward_batch, **kwargs):
        # memory monitoring
        if torch.cuda.is_available():
            mem_before_post = torch.cuda.memory_allocated() / 1024**3
        
        hidden_states, residual = CommunicateSummableTensorPairFn.execute(
            hidden_states_input_mode=afd_splitter_scatter_mode,
            residual_input_mode=afd_splitter_scatter_mode,
            output_mode=layer_input_scatter_mode,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=context,
        )
        
        return dict(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            **kwargs,
        )

    result = [_post_transform(**inputs) for inputs in inputs_arr]
    
    return result

def model_forward_afd(
    layers,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_data_scatter_mode: ScatterMode,
):
    num_layers = len(layers)
    m_stage = get_afd_mirco_batch()

    input_arrs = model_forward_afd_split_inputs(
        layers=layers,
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        input_data_scatter_mode=input_data_scatter_mode
    )

    stage_outputs: Dict[AFDForwardStage, deque[dict[Any, Any]]] = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: deque(),
        AFDForwardStage.AFD_FORWARD_STAGE_F: deque(),
    }

    stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].extend(input_arrs)

    def forward_A(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
        with nvtx_range(f"afd.layer{layer_id}.micro{mirco_batch_idx}.A(attn)"):
            hidden_states, residual = layers[layer_id].forward_afd_A(
                input_arrs[mirco_batch_idx]["positions"],
                inputs_args["hidden_states"],
                input_arrs[mirco_batch_idx]["forward_batch"],
                inputs_args["residual"],
            )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(
            dict (
            hidden_states = hidden_states,
            residual = residual,
        ))

    def forward_F(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        with nvtx_range(f"afd.layer{layer_id}.micro{mirco_batch_idx}.F(ffn)"):
            hidden_states, residual = layers[layer_id].forward_afd_F(
                inputs_args["hidden_states"],
                input_arrs[mirco_batch_idx]["forward_batch"],
                inputs_args["residual"],
            )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].append(
            dict (
            hidden_states = hidden_states,
            residual = residual,
        ))

    stage_executors = {
        AFDForwardStage.AFD_FORWARD_STAGE_A : forward_A,
        AFDForwardStage.AFD_FORWARD_STAGE_F : forward_F,
    }

    pipeline_stages = (
        AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
        if afd_is_attn()
        else AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    )

    for stage in pipeline_stages:
        type, *args = stage
        stage_executors.get(type)(*args)

    try:
        results = [stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft() for _ in range(m_stage)]
    except IndexError:
        raise ValueError("model_forward_afd: impossible path, a potential implementation bug?")

    all_hidden_states, all_residual = zip(
        *((res["hidden_states"], res["residual"]) for res in results)
    )
    
    cat_hidden_states = torch.cat(all_hidden_states, dim=0)
    cat_residual = torch.cat(all_residual, dim=0) if afd_is_attn() else None


    return (
        cat_hidden_states,
        cat_residual,
    )

def get_ffn_parallel_info():
    enable_ep_moe = global_server_args_dict.get("enable_ep_moe", False)
    enable_deepep_moe = global_server_args_dict.get("enable_deepep_moe", False)
    is_ep_mode = enable_ep_moe or enable_deepep_moe
    
    tp_group = get_tp_group()
    tp_rank = get_tensor_model_parallel_rank()
    
    if is_ep_mode:
        actual_tp_world_size = get_tensor_model_parallel_world_size()
        ep_size_from_config = global_server_args_dict.get("ep_size", 1)
        
        parallel_size = actual_tp_world_size

        parallel_rank = tp_rank
        
        if parallel_size == 1 and ep_size_from_config > 1:
            logger.warning(
                f"get_ffn_parallel_info: EP mode enabled with ep_size={ep_size_from_config} from config, "
                f"but actual tp_world_size={actual_tp_world_size}. "
                f"For EP to work with ep_size={ep_size_from_config}, you need to set --tp-size {ep_size_from_config} "
                f"or use {ep_size_from_config} GPUs/processes. "
                f"Currently using parallel_size={parallel_size} (single GPU/process)."
            )
    else:
        parallel_size = get_tensor_model_parallel_world_size()
        parallel_rank = tp_rank
    
    return parallel_rank, parallel_size, tp_group, is_ep_mode

class AFDCommunicator(LayerCommunicator):
    def __init__(self, layer_communicator: LayerCommunicator, perspective: AFDPerspective, layer_id: int):
        self.perspective = perspective
        self.layer_communicator = layer_communicator
        self.layer_id = layer_id
        pass
    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # just pass through
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            return hidden_states, residual

        # scatter hidden_states to all TP ranks
        result = self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)

        return result

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        lid = self.layer_id
        with nvtx_range(f"afd.L{lid}.prepare_mlp"):
            if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
                with nvtx_range(f"afd.L{lid}.prepare_mlp.ffn"):
                    parallel_rank, parallel_size, parallel_group, is_ep_mode = get_ffn_parallel_info()
                    if parallel_size > 1:

                        if parallel_rank == 0:
                            with nvtx_range(f"afd.L{lid}.prepare_mlp.ffn.recv_shape"):
                                hidden_states = get_tensor_communicator().recv_tensor()

                                # After receive, broadcast shape info to other ranks
                                if parallel_size > 1:
                                    # Convert dtype to string and remove 'torch.' prefix to get 'bfloat16' instead of 'torch.bfloat16'
                                    dtype_str = str(hidden_states.dtype)
                                    dtype_name = dtype_str.replace('torch.', '')
                                    shape_info = (hidden_states.shape[0], hidden_states.shape[1], dtype_name, hidden_states.device.index if hidden_states.device.type == 'cuda' else 0)
                                    # broadcast_object_list is also a collective operation, but it doesn't require barrier
                                    # because it's part of the synchronization protocol (sender waits for receivers)
                                    dist.broadcast_object_list([shape_info], src=parallel_group.ranks[0], group=parallel_group.device_group)
                        else:
                            with nvtx_range(f"afd.L{lid}.prepare_mlp.ffn.recv_shape"):
                                # Other ranks receive shape info, then create same-size tensor
                                shape_info_list = [None]
                                dist.broadcast_object_list(shape_info_list, src=parallel_group.ranks[0], group=parallel_group.device_group)
                                shape_info = shape_info_list[0]
                                num_tokens, hidden_dim, dtype_name, _ = shape_info
                                dtype = getattr(torch, dtype_name)
                                # Use current process device (each parallel rank has its own device)
                                current_device_idx = torch.cuda.current_device()
                                hidden_states = torch.empty(
                                    (num_tokens, hidden_dim),
                                    dtype=dtype,
                                    device=f"cuda:{current_device_idx}"
                                )

                        if parallel_size > 1 and hidden_states.shape[0] > 0:
                            with nvtx_range(f"afd.L{lid}.prepare_mlp.ffn.tp_broadcast"):
                                # Add barrier to ensure all ranks are ready before broadcast
                                # Use cpu_group for barrier to avoid device context issues (see GroupCoordinator.barrier)
                                torch.distributed.barrier(group=parallel_group.cpu_group)
                                # src should be the global rank (see GroupCoordinator.broadcast which uses self.ranks[src])
                                src_global_rank = parallel_group.ranks[0]
                                torch.distributed.broadcast(
                                    hidden_states,
                                    src=src_global_rank,
                                    group=parallel_group.device_group
                                )
                                torch.cuda.synchronize()

                        mlp_mode = self.layer_communicator.layer_scatter_modes.mlp_mode
                        if mlp_mode == ScatterMode.SCATTERED:
                            with nvtx_range(f"afd.L{lid}.prepare_mlp.ffn.scatter_split"):
                                if parallel_size > 1 and hidden_states.shape[0] > 0:
                                    hidden_states_list = list(hidden_states.tensor_split(parallel_size, dim=0))
                                    hidden_states = hidden_states_list[parallel_rank]
                                    if residual is not None and residual.shape[0] > 0:
                                        residual_list = list(residual.tensor_split(parallel_size, dim=0))
                                        residual = residual_list[parallel_rank]
                    else:
                        hidden_states = get_tensor_communicator().recv_tensor()
                return hidden_states, residual

            with nvtx_range(f"afd.L{lid}.prepare_mlp.attn"):
                mlp_mode = self.layer_communicator.layer_scatter_modes.mlp_mode
                attn_tp_rank = get_attention_tp_rank()
                attn_tp_size = get_attention_tp_size()
                context = self.layer_communicator._context

                if attn_tp_size > 1:
                    if mlp_mode == ScatterMode.FULL:
                        with nvtx_range(f"afd.L{lid}.prepare_mlp.attn.full_prepare"):
                            hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
                        if attn_tp_rank == 0:
                            with nvtx_range(f"afd.L{lid}.prepare_mlp.attn.send_hidden"):
                                get_tensor_communicator().send_tensor(hidden_states)
                    else:
                        with nvtx_range(f"afd.L{lid}.prepare_mlp.attn.scatter_prepare"):
                            residual_input_mode = self.layer_communicator.layer_scatter_modes.middle_residual_mode
                            if residual_input_mode == ScatterMode.SCATTERED and attn_tp_size > 1:
                                residual, local_residual = (
                                    forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]].clone(),
                                    residual,
                                )
                                attn_tp_all_gather_into_tensor(residual, local_residual)

                            if context.attn_dp_size != 1:
                                logger.error(format_afd_debug(attn_tp_rank, "AFDCommunicator.prepare_mlp (Attn): DP attention not supported in SCATTERED mode"))
                                if attn_tp_rank == 0:
                                    hidden_states += residual
                                raise NotImplementedError("SCATTERED mode with DP attention not yet supported in AFD")
                            else:
                                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                                hidden_states, residual = self.layer_communicator.post_attention_layernorm(hidden_states, residual)
                        if attn_tp_rank == 0:
                            with nvtx_range(f"afd.L{lid}.prepare_mlp.attn.send_hidden"):
                                get_tensor_communicator().send_tensor(hidden_states)
                else:
                    hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
                    get_tensor_communicator().send_tensor(hidden_states)
            return hidden_states, residual

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        lid = self.layer_id
        with nvtx_range(f"afd.L{lid}.postprocess_layer"):
            if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
                with nvtx_range(f"afd.L{lid}.postprocess_layer.ffn"):
                    mlp_mode = self.layer_communicator.layer_scatter_modes.mlp_mode
                    parallel_rank, parallel_size, parallel_group, is_ep_mode = get_ffn_parallel_info()
                    if parallel_size > 1:
                        if mlp_mode == ScatterMode.FULL:
                            if parallel_rank == 0:
                                with nvtx_range(f"afd.L{lid}.postprocess_layer.ffn.send"):
                                    # torch.cuda.synchronize()
                                    get_tensor_communicator().send_tensor(hidden_states)
                        else:
                            with nvtx_range(f"afd.L{lid}.postprocess_layer.ffn.merge_ag"):
                                hidden_states += residual
                                residual = None
                                if parallel_size > 1 and hidden_states.shape[0] > 0:
                                    hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

                            if parallel_rank == 0:
                                with nvtx_range(f"afd.L{lid}.postprocess_layer.ffn.send"):
                                    torch.cuda.synchronize()
                                    get_tensor_communicator().send_tensor(hidden_states)
                    else:
                        get_tensor_communicator().send_tensor(hidden_states)
                return hidden_states, residual

            with nvtx_range(f"afd.L{lid}.postprocess_layer.attn"):
                attn_tp_rank = get_attention_tp_rank()
                attn_tp_size = get_attention_tp_size()
                attn_tp_group = get_attention_tp_group()
                if attn_tp_size > 1:
                    if attn_tp_rank == 0:
                        with nvtx_range(f"afd.L{lid}.postprocess_layer.attn.recv"):
                            hidden_states = get_tensor_communicator().recv_tensor()
                    else:
                        with nvtx_range(f"afd.L{lid}.postprocess_layer.attn.alloc_recv_buf"):
                            if forward_batch.gathered_buffer is not None and forward_batch.gathered_buffer.shape[0] > 0:
                                num_tokens = forward_batch.input_ids.shape[0]
                                hidden_dim = forward_batch.gathered_buffer.shape[1]
                                hidden_states = torch.empty(
                                    (num_tokens, hidden_dim),
                                    dtype=forward_batch.gathered_buffer.dtype,
                                    device=forward_batch.gathered_buffer.device
                                )
                            else:
                                if residual is not None and residual.shape[0] > 0:
                                    hidden_states = torch.empty_like(residual)
                                else:
                                    logger.error(
                                        format_afd_debug(attn_tp_rank,
                                            f"AFDCommunicator.postprocess_layer (Attn): Cannot determine shape, "
                                            f"gathered_buffer={forward_batch.gathered_buffer is not None}, residual={residual is not None}"
                                        )
                                    )
                                    raise RuntimeError("Cannot determine hidden_states shape in AFD postprocess_layer")

                    if attn_tp_size > 1 and hidden_states.shape[0] > 0:
                        with nvtx_range(f"afd.L{lid}.postprocess_layer.attn.tp_broadcast"):
                            # Add barrier to ensure all ranks are ready before broadcast
                            # Use cpu_group for barrier to avoid device context issues (see GroupCoordinator.barrier)
                            torch.distributed.barrier(group=attn_tp_group.cpu_group)
                            # src should be the global rank (see GroupCoordinator.broadcast which uses self.ranks[src])
                            src_global_rank = attn_tp_group.ranks[0]
                            torch.distributed.broadcast(
                                hidden_states,
                                src=src_global_rank,
                                group=attn_tp_group.device_group
                            )

                    with nvtx_range(f"afd.L{lid}.postprocess_layer.attn.base_post"):
                        hidden_states, residual = self.layer_communicator.postprocess_layer(
                            hidden_states, residual, forward_batch
                        )
                else:
                    hidden_states = get_tensor_communicator().recv_tensor()

                    hidden_states, residual = self.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)

            return hidden_states, residual

class AFDProxyAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch) -> torch.Tensor:
        return hidden_states

class AFDProxyMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                forward_batch: Optional[ForwardBatch] = None) -> torch.Tensor:
        return hidden_states
