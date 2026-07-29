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
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
"""Inference-only Mixtral model."""

import logging
import os
from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn
from transformers import MixtralConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.afd import (
    AFDCommunicator,
    AFDProxyAttention,
    AFDProxyMLP,
    afd_is_ffn,
    get_afd_perspective,
    model_forward_afd,
)
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes, ScatterMode
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.nvtx_utils import format_batch_size_info, nvtx_range
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)

# Set to 1/true/yes to run the pre-LayerCommunicator path (forward_former) for A/B tests.
_MIXTRAL_FORWARD_FORMER_ENV = "SGLANG_MIXTRAL_USE_FORWARD_FORMER"
_mixtral_forward_former_warned = False


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.topk = TopK(
            top_k=top_k,
            renormalize=True,
        )

        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        self.experts = MoEImpl(
            num_experts=num_experts,
            top_k=top_k,
            layer_id=layer_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=add_prefix("experts", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(orig_shape)


class MixtralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        afd_perspective = get_afd_perspective()

        if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            self.self_attn = AFDProxyAttention()
        else:
            self.self_attn = MixtralAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                max_position=config.max_position_embeddings,
                num_kv_heads=config.num_key_value_heads,
                layer_id=layer_id,
                rope_theta=rope_theta,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )

        if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            self.block_sparse_moe = AFDProxyMLP()
        else:
            self.block_sparse_moe = MixtralMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("block_sparse_moe", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_id = layer_id
        # Every Mixtral decoder layer is sparse MoE (same convention as Qwen3 MoE).
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=True,
            is_previous_layer_sparse=True,
        )
        base_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
        if afd_perspective is not None:
            self.layer_communicator = AFDCommunicator(
                layer_communicator=base_communicator,
                perspective=afd_perspective,
                layer_id=layer_id,
            )
        else:
            self.layer_communicator = base_communicator

        self.use_forward_former = _env_truthy(_MIXTRAL_FORWARD_FORMER_ENV)
        if afd_perspective is not None:
            self.use_forward_former = False
        if self.use_forward_former:
            global _mixtral_forward_former_warned
            if not _mixtral_forward_former_warned:
                _mixtral_forward_former_warned = True
                logger.warning(
                    "%s is enabled: MixtralDecoderLayer uses forward_former "
                    "(legacy path) instead of LayerCommunicator.",
                    _MIXTRAL_FORWARD_FORMER_ENV,
                )

    def forward_former(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pre-LayerCommunicator implementation kept for parity checks and debugging."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual

    def forward_afd_A(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = forward_batch.batch_size
        num_tokens = forward_batch.seq_lens_sum
        hidden_size = (
            hidden_states.shape[-1]
            if hidden_states.shape[0] > 0
            else self.hidden_size
        )
        size_info = format_batch_size_info(
            batch_size=batch_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
        )
        with nvtx_range("afd_prepare_attn"):
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states, residual, forward_batch
            )
        if hidden_states.shape[0] != 0:
            with nvtx_range(f"afd_attn{size_info}"):
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )
        with nvtx_range("afd_prepare_mlp"):
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual

    def forward_afd_F(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = forward_batch.batch_size
        num_tokens = forward_batch.seq_lens_sum
        hidden_size = (
            hidden_states.shape[-1]
            if hidden_states.shape[0] > 0
            else self.hidden_size
        )
        size_info = format_batch_size_info(
            batch_size=batch_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
        )
        with nvtx_range(f"afd_ffn{size_info}"):
            hidden_states = self.block_sparse_moe(hidden_states)
        with nvtx_range("afd_postprocess_layer"):
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_forward_former:
            return self.forward_former(
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class MixtralModel(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: MixtralDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="layers",
            return_tuple=True,
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        if forward_batch.can_run_afd_overlap:
            hidden_states, residual = model_forward_afd(
                layers=self.layers,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=ScatterMode.model_input_output(),
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            # FFN AFD path: model_forward_afd returns residual=None; RMSNorm then returns a
            # single tensor — do not unpack as (h, _) or Python iterates the token dimension.
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if afd_is_ffn():
            return hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        afd_perspective = get_afd_perspective()

        if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            stacked_params_mapping = []
        else:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
            ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            expert_params_mapping = []
        else:
            expert_params_mapping = MoEImpl.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_local_experts,
            )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            # ATTN worker has AFDProxyMLP (no MoE params); skip MoE tensors silently.
            if (
                afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
                and "block_sparse_moe" in name
            ):
                continue
            # FFN worker has AFDProxyAttention; skip attention tensors silently.
            if (
                afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN
                and ".self_attn." in name
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    if name is None:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = MixtralForCausalLM
