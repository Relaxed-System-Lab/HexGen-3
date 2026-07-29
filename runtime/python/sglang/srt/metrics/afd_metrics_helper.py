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
Helper module for AFD metrics collection.

Provides a way for model layers to update metrics counters in scheduler.
"""

import threading
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Thread-local storage for scheduler instance
_thread_local = threading.local()


def set_scheduler_for_metrics(scheduler):
    """Set scheduler instance for current thread."""
    _thread_local.scheduler = scheduler


def get_scheduler_for_metrics():
    """Get scheduler instance for current thread."""
    return getattr(_thread_local, 'scheduler', None)


def update_attn_kv_ops(num_ops):
    """Update attention KV operations counter."""
    # Convert to Python native type if it's a Tensor
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
    
    num_ops = to_python_type(num_ops)
    
    scheduler = get_scheduler_for_metrics()
    if scheduler and hasattr(scheduler, 'attn_kv_ops_total'):
        scheduler.attn_kv_ops_total += num_ops

