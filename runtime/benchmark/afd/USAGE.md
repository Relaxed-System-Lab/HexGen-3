# AFD Metrics Collection - Detailed Usage Guide

## Environment Variables

### Server-side (SGLang)

- `SGLANG_ENABLE_AFD_METRICS=1`: Enable AFD metrics collection
- `SGLANG_AFD_METRICS_DIR`: Directory to store metrics JSONL files (default: `/tmp/sglang_afd_metrics`)
- `SGLANG_AFD_METRICS_INTERVAL`: Collection interval in seconds (default: `1.0`)

### Output Files

Server-side metrics are written to JSONL files:
- Format: `afd_metrics_{tp_rank}_{worker_type}.jsonl`
- Example: `afd_metrics_0_attn.jsonl`, `afd_metrics_0_ffn.jsonl`

Each line contains a JSON object with metrics for one time point:
```json
{
  "timestamp": 1234567890.123,
  "worker_type": "attn",
  "workload_uncached_tokens": 1000,
  "prefill_tps": 1200.0,
  "ffn_tps": 0.0,
  "attn_tps": 1200.0,
  "attn_kv_ops_per_sec": 115200.0,
  "gpu_utilization": 85.5,
  "avg_queue_wait_time": 0.012
}
```

## Metrics Collected

### Server-side (from SGLang)

1. **Workload**: KV cache uncached token count
2. **Throughput**:
   - Prefill TPS (tokens per second)
   - FFN TPS (FFN worker only)
   - Attn TPS (Attn worker only)
   - Attn KV-ops/sec (Attn worker only)
3. **Hardware**: GPU utilization (average across GPUs)
4. **Queue**: Average queue wait time

### Client-side (from benchmark script)

1. **TTFT**: Time to first token
2. **TBT**: Time between tokens (average)

## Time Alignment

The analysis script aligns metrics using time windows (default: 1 second):

1. Server metrics are collected at regular intervals (e.g., every 1 second)
2. Client metrics (TTFT/TBT) are recorded with request `created_time`
3. The analysis script groups client metrics into time windows based on `created_time`
4. Metrics are aligned by matching time windows

## Workflow Example

### Complete Workflow

```bash
# 1. Set environment variables
export SGLANG_ENABLE_AFD_METRICS=1
export SGLANG_AFD_METRICS_DIR=/tmp/sglang_afd_metrics
export SGLANG_AFD_METRICS_INTERVAL=1.0

# 2. Start Attn worker (terminal 1)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sglang.launch_server \
    --model-path <model> \
    --afd-perspective attn \
    --afd-mirco-batch 3 \
    --port 30000

# 3. Start FFN worker (terminal 2)
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m sglang.launch_server \
    --model-path <model> \
    --skip-server-warmup \
    --afd-perspective ffn \
    --afd-mirco-batch 3 \
    --port 30001

# 4. Run benchmark and analysis (terminal 3)
# The script will automatically collect client metrics, then analyze and visualize results
python benchmark/afd/bench_afd_with_metrics.py \
    --api-url http://localhost:30000 \
    --num-requests 200 \
    --request-rate 2.0 \
    --server-metrics-dir /tmp/sglang_afd_metrics \
    --output-dir results \
    --plot-output metrics_analysis.png
```

**Note**: If `--server-metrics-dir` is not provided, the script will only collect client metrics without analysis.

## Output Files

### From Benchmark Script

- `client_metrics.json`: Client-side metrics (TTFT/TBT) - saved in current directory
- `results/metrics_analysis.png`: Visualization plots (if `--server-metrics-dir` provided)
- `results/report.txt`: Text report with correlations (if `--server-metrics-dir` provided)
- `results/aligned_metrics.json`: Aligned metrics data (if `--server-metrics-dir` provided)

### Understanding the Results

The correlation analysis shows which metrics best track workload changes:
- High positive correlation (>0.7): Metric increases with workload
- High negative correlation (<-0.7): Metric decreases with workload (e.g., TPS might decrease under high load)
- Low correlation (<0.3): Metric doesn't track workload well

The metric with the highest absolute correlation with workload is the best indicator for autoscaling decisions.

