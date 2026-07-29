# adapted from https://github.com/hahnyuan/LLM-Viewer/blob/main/model_analyzer.py
import math
from transformers import AutoConfig
from simulator.configs.hardware import hardware_params
from .roofline import roofline_analyze



ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]

# Buckets for component-level decode time attribution (used by AFD analyzer)
ATTENTION_LAYER_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fused_attention",
    "attn_add",
    "attn_norm",
}

FFN_LAYER_NAMES = {
    "gate_proj",
    "up_proj",
    "down_proj",
    "mlp_act",
    "mlp_add",
    "mlp_norm",
}


class ModelAnalyzer:
    def __init__(self, model_id, config, hardware):
        self.model_id = model_id
        self.hardware = hardware
        try:
            self.model_params = AutoConfig.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception:
            # Keep offline scheduling/evaluation usable when Hugging Face config
            # downloads are unavailable.
            from transformers.models.llama.configuration_llama import LlamaConfig

            if "qwen3-30b-a3b" in model_id.lower():
                # Qwen3-30B-A3B public config summary. The copied analyzer has
                # dense-Llama kernels, so MoE fields are carried for schedulers
                # while dense linear-layer estimates use intermediate_size.
                self.model_params = LlamaConfig(
                    hidden_size=2048,
                    intermediate_size=6144,
                    num_hidden_layers=48,
                    num_attention_heads=32,
                    num_key_value_heads=4,
                    vocab_size=151936,
                    max_position_embeddings=40960,
                )
                self.model_params.num_experts = 128
                self.model_params.num_experts_per_tok = 8
                self.model_params.moe_intermediate_size = 768
            else:
                # Values match Llama-3.1-8B closely enough for comparative
                # simulator sweeps when no model-specific fallback is known.
                self.model_params = LlamaConfig(
                    hidden_size=4096,
                    intermediate_size=14336,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=8,
                    vocab_size=128256,
                    rope_scaling={"type": "dynamic", "factor": 8.0},
                )
        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None
        self.config = config
        
    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        memory_access = (
            load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        )
        arithmetic_intensity, performance, bound = roofline_analyze(
            bandwidth, max_OPS, OPs, memory_access
        )
        # Handle case when OPs = 0 (component is disabled)
        if OPs == 0 or performance == 0:
            inference_time = 0
        else:
            inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }
            

    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        tp_size: int = 1,
        disable_attention: bool = False,
        disable_ffn: bool = False,
    ):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        tp_size: tensor parallel size

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        self.results = {
            "decode": {},
            "prefill": {}
        }
        if kv_bit is None:
            kv_bit = a_bit
        
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size
        
        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        model_params = self.model_params
        num_attention_heads = self.config.get_num_attention_heads(model_params)
        hidden_size = self.config.get_hidden_size(model_params)
        num_key_value_heads = self.config.get_num_key_value_heads(model_params)
        num_hidden_layers = self.config.get_num_hidden_layers(model_params)

        for name, (ic, oc) in self.config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            
            # Skip FFN layers if disable_ffn is True
            is_ffn_layer = name in ["gate_proj", "up_proj", "down_proj"]
            if disable_ffn and is_ffn_layer:
                ops = 0
            else:
                ops = ic * oc * batchsize * 2
            
            self._analyze_to_results(
                "decode",
                name,
                OPs=ops,
                load_weight=ic * oc * w_byte if not (disable_ffn and is_ffn_layer) else 0,
                load_act=ic * batchsize * a_byte if not (disable_ffn and is_ffn_layer) else 0,
                store_act=0 if is_kv_proj else (oc * batchsize * a_byte if not (disable_ffn and is_ffn_layer) else 0),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            if disable_ffn and is_ffn_layer:
                ops_prefill = 0
            else:
                ops_prefill = ic * oc * batchsize * seqlen * 2
            
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ops_prefill,
                load_weight=ic * oc * w_byte if not (disable_ffn and is_ffn_layer) else 0,
                load_act=ic * batchsize * seqlen * a_byte if not (disable_ffn and is_ffn_layer) else 0,
                store_act=(0 if is_kv_proj else (oc * batchsize * seqlen * a_byte if not (disable_ffn and is_ffn_layer) else 0)),
                load_kv_cache=0,
                store_kv_cache=(
                    0 if is_normal_proj else oc * batchsize * seqlen * kv_byte
                ),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        
        # Disable attention if requested
        if disable_attention:
            attention_ops = 0
        else:
            attention_ops = qk_matmul_OPs + sv_matmul_OPs + softmax_OPs
        
        name: str = f"fused_attention"
        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        block_size_r = min(
            math.ceil(onchip_buffer / (kv_byte * head_size)), head_size
        )
        n_blocks_r = math.ceil(1 / block_size_r)
        q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
        o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
        self._analyze_to_results(
            "decode",
            name,
            OPs=attention_ops,
            load_weight=0,
            load_act=q_numel if not disable_attention else 0,
            store_act=(o_numel * 2) if not disable_attention else 0,  # initialize O and save O
            load_kv_cache=n_blocks_r
            * (seqlen)
            * head_size
            * batchsize
            * num_key_value_heads
            * kv_byte
            * 2 if not disable_attention else 0,
            store_kv_cache=0,
        )

        for name in self.config.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 2 if not disable_ffn else 0,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte * 2 if not disable_ffn else 0,
                store_act=batchsize * hidden_size * 1 * a_byte if not disable_ffn else 0,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = (
            seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        )
        sv_matmul_OPs = (
            seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        )
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        name = f"fused_attention"
        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
        block_size_r = min(
            math.ceil(onchip_buffer / (kv_byte * head_size)), head_size
        )
        n_blocks_r = math.ceil(seqlen / block_size_r)
        q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
        o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
        
        # Disable attention if requested
        if disable_attention:
            prefill_attention_ops = 0
        else:
            prefill_attention_ops = qk_matmul_OPs + sv_matmul_OPs + softmax_OPs
        
        self._analyze_to_results(
            "prefill",
            name,
            OPs=prefill_attention_ops,
            load_weight=0,
            load_act=q_numel if not disable_attention else 0,
            store_act=(o_numel * 2) if not disable_attention else 0,  # initialize O and save O
            load_kv_cache=n_blocks_r
            * (seqlen)
            * head_size
            * batchsize
            * num_key_value_heads
            * kv_byte
            * 2 if not disable_attention else 0,
            store_kv_cache=0,
        )
        
        for name in self.config.get_norm_layers(model_params):
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1 * 2 if not disable_ffn else 0,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte * 2 if not disable_ffn else 0,
                store_act=batchsize * hidden_size * seqlen * a_byte if not disable_ffn else 0,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += (
                        result[data_name] * num_hidden_layers
                    )

        # memory footprint
        weight_kv_footprint = (
            total_results["prefill"]["load_weight"]
            + total_results["prefill"]["store_kv_cache"]
        )
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = (
            decode_tmp_act + weight_kv_footprint
        )
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"][
            "load_weight"
        ]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results[
            "prefill"
        ]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = (
            prefill_tmp_act + weight_kv_footprint
        )
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results[
            "prefill"
        ]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results[
            "prefill"
        ]["store_kv_cache"]

        name = "lm_head"
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[
                    layer_info["stage"]
                ][layer_info["name"]][data_name]

        self.results["total_results"] = total_results
        return self.results

    def analyze_decode_breakdown(
        self,
        seqlen: int,
        batchsize: int,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = None,
        tp_size: int = 1,
        disable_attention: bool = False,
        disable_ffn: bool = False,
    ) -> dict:
        """
        Return a lightweight decode-only breakdown separating attention-side and FFN-side
        compute for one token at a given sequence length / batch size.

        The numbers returned are **per layer** times; callers should multiply by the
        model depth when estimating a full decode pass.
        """
        res = self.analyze(
            seqlen=seqlen,
            batchsize=batchsize,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            tp_size=tp_size,
            disable_attention=disable_attention,
            disable_ffn=disable_ffn,
        )
        decode_components = res.get("decode", {})

        def _sum(names):
            return sum(
                decode_components.get(name, {}).get("inference_time", 0.0)
                for name in names
            )

        attention_time = _sum(ATTENTION_LAYER_NAMES)
        ffn_time = _sum(FFN_LAYER_NAMES)
        num_layers = self.config.get_num_hidden_layers(self.model_params)

        return {
            "per_layer": {
                "attention_s": attention_time,
                "ffn_s": ffn_time,
                "total_s": attention_time + ffn_time,
            },
            "num_layers": num_layers,
            "raw": res,
        }

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        tp_size: int = 1,
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            tp_size,
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"][
            "inference_time"
        ]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(
                i,
                batchsize,
                w_bit,
                a_bit,
                kv_bit,
                tp_size,
            )
            inference_time += result["total_results"]["decode"]["inference_time"]
            tpot = result["total_results"]["decode"]["inference_time"]
        return {
            "inference_time": inference_time, 
            "prefill_time": prefill_time,
            "tpot": inference_time / gen_len,
        }

    def get_hardware_info(self):
        bandwidth = hardware_params[self.hardware]["bandwidth"]
        if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_OPS = hardware_params[self.hardware]["INT8"]
        else:
            max_OPS = hardware_params[self.hardware]["FP16"]
        onchip_buffer = hardware_params[self.hardware]["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(
            self.model_params
        ) != self.config.get_num_key_value_heads(self.model_params):
            GQA = True
        else:
            GQA = False
        info = {"GQA": GQA}  # group query attention
        return info
