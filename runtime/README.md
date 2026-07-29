# HexGen-3 Runtime

This directory contains the SGLang-based runtime used by HexGen-3. The runtime
can be launched manually for debugging or controlled by the live autoscaler from
the top-level README.

For environment setup, StepMesh build instructions, and the live autoscaling
workflow, see `../README.md`.

## Manual AFD Launch

The single-node example below starts a small AFD deployment:

- 1 GPU for prefill
- 2 GPUs for decode-attention (`--tp 2`)
- 1 GPU for decode-FFN
- 1 mini load balancer on port `30000`

Run each process in a separate shell and set these cluster-specific values in
every shell:

```bash
MODEL_PATH="/path/to/model"
AFD_HOST="<node IP address reachable by the attention and FFN workers>"
STEPMESH_INTERFACE="<network interface used by StepMesh>"
RDMA_DEVICE="<RDMA device used for prefill-decode KV transfer>"
PREFILL_GPU="<prefill GPU ID>"
ATTENTION_GPUS="<two comma-separated attention GPU IDs>"
FFN_GPU="<FFN GPU ID>"
```

`STEPMESH_INTERFACE` is a network interface, while `RDMA_DEVICE` is the device
accepted by `--disaggregation-ib-device`; they need not have the same name.

### Prefill Worker

```bash
export CUDA_VISIBLE_DEVICES="$PREFILL_GPU"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-ib-device "$RDMA_DEVICE" \
  --disaggregation-bootstrap-port 9001 \
  --host 127.0.0.1 \
  --port 30001 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 65536
```

### Decode-Attention Worker

```bash
export CUDA_VISIBLE_DEVICES="$ATTENTION_GPUS"
export AFD_SCHED_HOST="$AFD_HOST"
export DMLC_PS_ROOT_URI="$AFD_HOST"
export DMLC_NODE_HOST="$AFD_HOST"
export MLC_INTERFACE="$STEPMESH_INTERFACE"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --disaggregation-mode decode \
  --afd-perspective attn \
  --disaggregation-ib-device "$RDMA_DEVICE" \
  --disaggregation-bootstrap-port 9001 \
  --afd-mirco-batch 2 \
  --port 30002 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 65536 \
  --tp 2
```

### Decode-FFN Worker

```bash
export CUDA_VISIBLE_DEVICES="$FFN_GPU"
export AFD_SCHED_HOST="$AFD_HOST"
export DMLC_PS_ROOT_URI="$AFD_HOST"
export DMLC_NODE_HOST="$AFD_HOST"
export MLC_INTERFACE="$STEPMESH_INTERFACE"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --disaggregation-mode null \
  --skip-server-warmup \
  --afd-perspective ffn \
  --afd-mirco-batch 2 \
  --port 30003 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 65536 \
  --mem-fraction-static 0.75
```

### Mini Load Balancer

```bash
python3 -m sglang.srt.disaggregation.mini_lb \
  --prefill http://127.0.0.1:30001 \
  --decode http://127.0.0.1:30002 \
  --host 127.0.0.1 \
  --port 30000
```

Requests should be sent to:

```text
http://127.0.0.1:30000/generate
```

## Verification

```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 100 --port 30000

python -m sglang.bench_serving \
  --port 30000 \
  --backend sglang \
  --dataset-name random-ids \
  --random-range-ratio 1 \
  --random-input-len 1280 \
  --random-output-len 1280 \
  --num-prompt 200
```
