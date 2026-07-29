import math
from typing import List
from typing import TYPE_CHECKING, Any
from transformers import AutoConfig
try:
    from humanize import naturalsize
except Exception:
    def naturalsize(value, binary=False):
        suffixes = ["B", "KB", "MB", "GB", "TB"]
        base = 1024.0 if binary else 1000.0
        size = float(value)
        idx = 0
        while abs(size) >= base and idx < len(suffixes) - 1:
            size /= base
            idx += 1
        return f"{size:.1f} {suffixes[idx]}"
from .config import _normalize_parallel_config

if TYPE_CHECKING:
    from .request import GenerationRequest

class MemoryPlanner:
    def __init__(
        self,
        model_params: AutoConfig,
        model_config: Any,
        hardware_params: dict,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        gpu_utilization: float = 0.9,
        parallel_config=None,
        block_size: int = 16,
    ):
        """
        Initialize memory planner for KV cache management.

        Args:
            model_params: HuggingFace model configuration object
            model_config: Model-specific configuration package
            hardware_params: Hardware parameter dictionary with memory and compute specs
            w_bit: Weight precision in bits
            a_bit: Activation precision in bits
            kv_bit: KV cache precision in bits
            gpu_utilization: Target GPU utilization (0.0-1.0)
            parallel_config: Optional tensor/pipeline parallel configuration
            block_size: Number of tokens per memory block
        """
        self.model_params = model_params
        self.parallel_config = _normalize_parallel_config(parallel_config)
        self._tp_size = self.parallel_config.tensor_parallel_size
        self._pp_size = self.parallel_config.pipeline_parallel_size
        self.hardware_params = hardware_params
        self.gpu_utilization = gpu_utilization
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.block_size = block_size
        self._allocated_blocks = 0
        self._allocation_map = {}
        self._config = model_config
        if self._config:
            self._block_memory_size_per_shard = self._calculate_block_memory_size_per_shard()
            self._max_num_blocks = self.get_max_num_blocks()
        else:
            raise ValueError("Model config is required for memory planning")
        
    def get_max_num_blocks(self):
        """
        Calculate the maximum number of KV cache blocks that can be allocated.

        Returns:
            int: Maximum number of blocks available for KV cache allocation
        """
        # TODO(xiaozhe): we ignored the memory for activations
        per_shard_memory = self.hardware_params["vmemory"]

        # memory for weights
        w_memory_total = self.get_weights_memory()
        w_memory_per_shard = w_memory_total / self._tp_size if self._tp_size else w_memory_total

        # Check if weights exceed per-shard memory
        if w_memory_per_shard >= per_shard_memory:
            print(
                "Warning: Model weights per shard "
                f"({naturalsize(w_memory_per_shard)}) exceed GPU memory "
                f"({naturalsize(per_shard_memory)}). No KV cache blocks available "
                f"even with tensor parallel size {self._tp_size}."
            )
            return 0

        total_block_memory_size_per_shard = self._block_memory_size_per_shard
        if total_block_memory_size_per_shard <= 0:
            return 0

        available_memory_per_shard = per_shard_memory - w_memory_per_shard
        if available_memory_per_shard <= 0:
            return 0

        max_blocks_per_shard = math.floor(
            available_memory_per_shard / total_block_memory_size_per_shard
        )

        # Ensure we don't return negative values
        return max(0, max_blocks_per_shard)

    def get_weights_memory(self):
        """
        Calculate the memory required to store model weights.

        Returns:
            float: Total memory in bytes required for all model weights
        """
        mlp_weights = (
            3
            * self._get_hidden_size()
            * self._get_intermediate_size()
            * self.w_bit
            / 8
        )

        q_weights = (
            self._get_hidden_size()
            * self._get_num_attention_heads()
            * self._get_head_dim()
        )

        kv_weights = (
            2
            * self._get_hidden_size()
            * self._get_head_dim()
            * self._get_num_key_value_heads()
        )

        o_weights = (
            self._get_hidden_size()
            * self._get_num_attention_heads()
            * self._get_head_dim()
        )
        self_attn_weights = (q_weights + kv_weights + o_weights) * self.w_bit / 8

        lm_head_weights = (
            self._get_hidden_size()
            * self._get_vocab_size()
            * self.w_bit
            / 8
        )
        embedding_weights = (
            self._get_hidden_size()
            * self._get_vocab_size()
            * self.w_bit
            / 8
        )

        return (
            (mlp_weights + self_attn_weights)
            * self._get_num_hidden_layers()
            + lm_head_weights
            + embedding_weights
        )

    def print_status(self):
        """
        Print current memory allocation status for debugging.
        """
        per_shard_memory = self.hardware_params["vmemory"]
        weights_per_shard = self.get_weights_memory_per_shard()
        print(
            "Weights memory per shard / GPU memory: "
            f"{naturalsize(weights_per_shard)} / {naturalsize(per_shard_memory)} "
            f"(tensor_parallel_size={self._tp_size})"
        )
        print(
            f"Allocated blocks/Total blocks: {self._allocated_blocks}/{self._max_num_blocks}"
        )

    def can_allocate_request(self, request: "GenerationRequest"):
        """
        Check if there is enough memory to allocate blocks for a request.

        Args:
            request: The GenerationRequest to check allocation for

        Returns:
            bool: True if allocation is possible, False otherwise
        """
        # If no blocks are available, return False immediately
        if self._max_num_blocks == 0:
            return False

        additional_blocks = self._estimate_required_blocks(request)
        if additional_blocks == 0:
            return True

        alloc_limit = math.floor(self._max_num_blocks * 0.95)
        if alloc_limit <= 0:
            return False

        return self._allocated_blocks + additional_blocks <= alloc_limit

    def allocate(self, request: "GenerationRequest"):
        """
        Allocate memory blocks for a request's KV cache.

        Args:
            request: The GenerationRequest to allocate memory for
        """

        def _allocate_blocks(request_id: str, num_blocks: int):
            if num_blocks == 0:
                return
            self._allocated_blocks += num_blocks
            if request_id not in self._allocation_map:
                self._allocation_map[request_id] = num_blocks
            else:
                self._allocation_map[request_id] += num_blocks
            assert (
                self._allocated_blocks <= self._max_num_blocks
            ), "Exceeding memory limit"

        additional_blocks = self._estimate_required_blocks(request)
        if additional_blocks == 0:
            return

        _allocate_blocks(request.req_id, additional_blocks)

    def free(self, request_ids: List[str]):
        """
        Free memory blocks allocated to completed requests.

        Args:
            request_ids: List of request IDs whose memory should be freed
        """
        for req_id in request_ids:
            num_blocks = self._allocation_map.pop(req_id, 0)
            self._allocated_blocks -= num_blocks
        assert self._allocated_blocks >= 0, "Negative allocated blocks"

    def calculate_model_memory(self) -> float:
        """
        Calculate the total memory required for model weights.

        Returns:
            float: Total memory in bytes required for model weights
        """
        return self.get_weights_memory()

    def usage(self) -> tuple[int, int]:
        """Return the current (used, total) block counts."""
        return self._allocated_blocks, self._max_num_blocks

    def reset_allocations(self):
        """Reset all KV-cache allocations tracked by this planner."""
        self._allocated_blocks = 0
        self._allocation_map.clear()

    def has_allocation(self, request_id: str) -> bool:
        """Check if a request currently has allocated blocks."""
        return request_id in self._allocation_map

    def get_weights_memory_per_shard(self) -> float:
        """Return weight memory required per tensor-parallel shard."""
        if self._tp_size == 0:
            return self.get_weights_memory()
        return self.get_weights_memory() / self._tp_size

    def get_allocated_block_count(self) -> int:
        """Return the number of KV blocks currently allocated."""
        return self._allocated_blocks

    def get_max_block_count(self) -> int:
        """Return the total number of KV blocks available per shard."""
        return self._max_num_blocks

    def get_allocated_kv_memory_per_shard(self) -> float:
        """Return KV cache memory currently allocated per shard."""
        return self._allocated_blocks * self._block_memory_size_per_shard

    def get_total_kv_memory_capacity_per_shard(self) -> float:
        """Return total KV cache memory capacity per shard."""
        return self._max_num_blocks * self._block_memory_size_per_shard

    def estimate_additional_kv_memory_per_shard(self, request: "GenerationRequest") -> float:
        """Estimate additional KV cache memory per shard required for the request."""
        additional_blocks = self._estimate_required_blocks(request)
        return additional_blocks * self._block_memory_size_per_shard

    def _estimate_required_blocks(self, request: "GenerationRequest") -> int:
        """Estimate how many additional blocks are required for the request."""
        current_tokens = max(1, request.input_length + request.generated_tokens)
        if request.req_id not in self._allocation_map:
            return math.ceil(current_tokens / self.block_size)

        num_tokens_reserved = self._allocation_map[request.req_id] * self.block_size
        num_tokens_required = max(0, current_tokens - num_tokens_reserved)
        if num_tokens_required == 0:
            return 0

        return math.ceil(num_tokens_required / self.block_size)

    def _calculate_block_memory_size_per_shard(self) -> float:
        """Compute memory footprint per KV cache block on a shard."""
        heads_per_shard = max(1, math.ceil(self._get_num_key_value_heads() / self._tp_size))
        bytes_per_element = self.kv_bit / 8
        per_layer = 2 * self.block_size * heads_per_shard * self._get_head_dim() * bytes_per_element
        return per_layer * self._get_num_hidden_layers()

    def _get_hidden_size(self) -> int:
        if self._config and hasattr(self._config, "get_hidden_size"):
            return self._config.get_hidden_size(self.model_params)
        return getattr(self.model_params, "hidden_size", 0)

    def _get_num_attention_heads(self) -> int:
        if self._config and hasattr(self._config, "get_num_attention_heads"):
            return self._config.get_num_attention_heads(self.model_params)
        return getattr(self.model_params, "num_attention_heads", 1)

    def _get_num_key_value_heads(self) -> int:
        if self._config and hasattr(self._config, "get_num_key_value_heads"):
            return self._config.get_num_key_value_heads(self.model_params)
        return getattr(self.model_params, "num_key_value_heads", self._get_num_attention_heads())

    def _get_head_dim(self) -> int:
        if self._config and hasattr(self._config, "get_head_dim"):
            return self._config.get_head_dim(self.model_params)
        num_heads = max(1, self._get_num_attention_heads())
        return self._get_hidden_size() // num_heads

    def _get_num_hidden_layers(self) -> int:
        if self._config and hasattr(self._config, "get_num_hidden_layers"):
            return self._config.get_num_hidden_layers(self.model_params)
        return getattr(self.model_params, "num_hidden_layers", 1)

    def _get_intermediate_size(self) -> int:
        if self._config and hasattr(self._config, "get_intermediate_size"):
            return self._config.get_intermediate_size(self.model_params)
        return getattr(self.model_params, "intermediate_size", self._get_hidden_size() * 4)

    def _get_vocab_size(self) -> int:
        if self._config and hasattr(self._config, "get_vocab_size"):
            return self._config.get_vocab_size(self.model_params)
        return getattr(self.model_params, "vocab_size", 0)
