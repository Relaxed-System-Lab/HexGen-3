# AFD Autoscaling Metrics Collection

This directory contains a complete system for evaluating AFD (Attention-FFN Disaggregation) autoscaling metrics to determine which metric best reflects workload changes.

## System Overview

The system consists of three main components:

1. **Server-side metrics collector** (integrated in SGLang): Collects TPS, KV-ops, GPU utilization, queue wait time, and workload metrics
2. **Client-side metrics collector** (`bench_afd_with_metrics.py`): Sends requests and collects TTFT/TBT
3. **Analysis tool** (`analyze_afd_metrics.py`): Aligns metrics by time, calculates correlations, and generates reports

## Architecture

```
┌─────────────────────────────────────────┐
│  SGLang Server (with AFD metrics)      │
│  - Collects metrics every 1s           │
│  - Writes to JSONL files               │
│  - Separate file per worker type       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Benchmark Script                       │
│  - Sends requests                       │
│  - Records TTFT/TBT with timestamps    │
│  - Outputs JSON file                    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Analysis Script                        │
│  - Aligns metrics by time windows      │
│  - Calculates correlations              │
│  - Generates plots and reports          │
└─────────────────────────────────────────┘
```

## Overview

The goal is to collect **real-time time-series metrics** aligned with **workload** (KV cache-missed prefill input) and evaluate which metric best correlates with workload fluctuations.

**Workload Definition**: KV cache-missed prefill input = `prompt_tokens_total - cached_tokens_total`
- Only requests that require actual computation (cache misses) represent true load
- Fully cached requests consume negligible computational resources

## Metrics Collected

### 1. Throughput Metrics
- **Prefill TPS**: Tokens-Per-Second for prefill workers (from prefill/unified worker)
- **FFN TPS**: Tokens-Per-Second for FFN workers (from FFN worker in AFD mode)
- **Attention KV-ops/sec**: KV-operations per second for attention workers (from attention worker in AFD mode)

### 2. Latency Metrics
- **TTFT**: Time-To-First-Token (from client measurements)
- **TBT**: Time-Between-Tokens (from client measurements)

### 3. Hardware Utilization Metrics
- **GPU utilization**: Average GPU utilization across all GPUs

### 4. Queue-based Metrics
- **Average queue wait time**: Average time requests wait in queue before processing

## Key Concepts

### Which Metrics to Collect?

1. **Prefill TPS**: Collect from **prefill worker** (or unified worker if not using AFD)
2. **FFN TPS**: Collect from **FFN worker** (only in AFD mode)
3. **Attention KV-ops/sec**: Collect from **attention worker** (only in AFD mode)
4. **TTFT/TBT**: Collect from **client-side** measurements during request processing
5. **GPU utilization**: Collect from **all workers** (average across GPUs)
6. **Queue wait time**: Collect from **all workers** (average)

### Time Alignment

All metrics are collected at **fixed intervals** (default: 1 second) with synchronized timestamps, ensuring:
- All metrics share the same time axis
- Easy correlation analysis between metrics and workload
- Real-time visualization of metric fluctuations

### Data Format

The output is **time-series data** (not aggregated statistics):
- Each sample contains: timestamp, workload, and all metrics
- Samples are collected at regular intervals (e.g., every 1 second)
- Suitable for plotting and correlation analysis

## Quick Start

### Step 1: Enable AFD Metrics Collection in SGLang

Set environment variables before starting SGLang servers:

```bash
export SGLANG_ENABLE_AFD_METRICS=1
export SGLANG_AFD_METRICS_DIR=/tmp/sglang_afd_metrics
export SGLANG_AFD_METRICS_INTERVAL=1.0  # Collection interval in seconds (default: 1.0)
```

### Step 2: Start SGLang AFD Servers

**Attn Worker:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>
export SGLANG_ENABLE_AFD_METRICS=1
export SGLANG_AFD_METRICS_DIR=/tmp/sglang_afd_metrics

python -m sglang.launch_server \
    --model-path <model> \
    --afd-perspective attn \
    --afd-mirco-batch 3 \
    --port 30000
```

**FFN Worker:**
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>
export SGLANG_ENABLE_AFD_METRICS=1
export SGLANG_AFD_METRICS_DIR=/tmp/sglang_afd_metrics

python -m sglang.launch_server \
    --model-path <model> \
    --skip-server-warmup \
    --afd-perspective ffn \
    --afd-mirco-batch 3 \
    --port 30001
```

### Step 3: Run Benchmark and Collect Client Metrics

In another terminal:

```bash
python benchmark/afd/bench_afd_with_metrics.py \
    --api-url http://localhost:30000 \
    --num-requests 100 \
    --request-rate 2.0 \
    --output client_metrics.json
```

### Step 4: Analyze Metrics

After the benchmark completes:

```bash
python benchmark/afd/analyze_afd_metrics.py \
    --server-metrics-dir /tmp/sglang_afd_metrics \
    --client-metrics client_metrics.json \
    --output-dir results \
    --plot-output metrics_analysis.png
```

The analysis script will:
- Load metrics from all workers
- Align server and client metrics by time windows
- Calculate correlations with workload
- Generate visualization plots
- Create a text report

## Usage Details

### Server-side Metrics Collection

#### For AFD Setup (Separate Attn and FFN Workers)

```bash
# Collect metrics from both attn and ffn workers
python benchmark/afd/collect_afd_metrics.py \
    --attn-metrics-url http://localhost:30000/metrics \
    --ffn-metrics-url http://localhost:30001/metrics \
    --duration 600 \
    --interval 1.0 \
    --output metrics.json \
    --plot metrics.png
```

#### For Unified/Non-AFD Setup

```bash
# Collect metrics from unified worker
python benchmark/afd/collect_afd_metrics.py \
    --metrics-url http://localhost:30000/metrics \
    --duration 600 \
    --interval 1.0 \
    --output metrics.json \
    --plot metrics.png
```

### Step 2: Extract Latency from Request Traces (Optional)

If you have real request traces from a benchmark run, extract latency data:

```bash
python benchmark/afd/extract_latency_from_requests.py \
    --input requests.jsonl \
    --output latency.json
```

Then include it in metrics collection:

```bash
python benchmark/afd/collect_afd_metrics.py \
    --attn-metrics-url http://localhost:30000/metrics \
    --ffn-metrics-url http://localhost:30001/metrics \
    --latency-file latency.json \
    --duration 600 \
    --interval 1.0 \
    --output metrics.json \
    --plot metrics.png
```

### Command Line Arguments for `collect_afd_metrics.py`

- `--metrics-url`: Metrics URL for unified worker (use with non-AFD setup)
- `--attn-metrics-url`: Metrics URL for attention worker (AFD mode)
- `--ffn-metrics-url`: Metrics URL for FFN worker (AFD mode)
- `--duration`: Collection duration in seconds (default: 300)
- `--interval`: Sampling interval in seconds (default: 1.0)
- `--output`: Output JSON file for time-series data (default: afd_metrics.json)
- `--plot`: Output plot file (default: afd_metrics.png)
- `--latency-file`: JSON file with latency samples (optional)

### Command Line Arguments for `extract_latency_from_requests.py`

- `--input`: Input file with request traces (JSONL or JSON)
- `--output`: Output JSON file with latency data (default: latency.json)

## Prerequisites

1. **SGLang server running with AFD enabled**
   - Attn worker: `python -m sglang.launch_server --afd-perspective attn ...`
   - FFN worker: `python -m sglang.launch_server --skip-server-warmup --afd-perspective ffn ...`

2. **Required Python packages**:
   ```bash
   pip install aiohttp numpy prometheus-client requests
   ```

3. **nvidia-smi** (for GPU utilization monitoring):
   - Should be available in PATH
   - Requires NVIDIA drivers

## Output Format

### JSON Output

The script outputs **time-series data** in JSON format:

```json
{
  "samples": [
    {
      "timestamp": 1234567890.123,
      "workload": 1234.56,
      "prefill_tps": 1200.0,
      "ffn_tps": 1150.0,
      "attn_kv_ops_per_sec": 110400.0,
      "ttft": 0.123,
      "tbt": 0.035,
      "gpu_utilization": 85.5,
      "avg_queue_wait_time": 0.012
    },
    ...
  ],
  "metadata": {
    "duration": 600,
    "interval": 1.0,
    "num_samples": 600,
    "collectors": ["attn", "ffn"]
  }
}
```

### Plot Output

The script generates a plot with:
1. **Throughput metrics** vs workload
2. **Latency metrics** vs workload
3. **GPU utilization** vs workload
4. **Queue wait time** vs workload
5. **Correlation scatter plots**
6. **Summary statistics** (correlation coefficients)

The plot helps visualize which metric best tracks workload changes.

## Metrics Collection Details

### Workload Calculation

**Workload = KV cache-missed prefill tokens per second**

Calculated as:
```
workload = (prompt_tokens_total - cached_tokens_total) / time_interval
```

- `prompt_tokens_total`: Counter from Prometheus (`sglang:prompt_tokens_total`)
- `cached_tokens_total`: Counter from Prometheus (`sglang:cached_tokens_total`)
- Rate calculated as difference between consecutive samples divided by time interval

### Throughput Metrics

- **Prefill TPS**: Rate of prefill tokens processed per second
  - Collected from prefill/unified worker
  - From `sglang:gen_throughput` or calculated from `sglang:prompt_tokens_total`
  
- **FFN TPS**: Rate of tokens processed by FFN workers per second
  - Collected from FFN worker (AFD mode only)
  - From `sglang:gen_throughput` on FFN worker
  
- **Attention KV-ops/sec**: Estimated KV cache operations per second
  - Collected from attention worker (AFD mode only)
  - Estimated as: `throughput * num_layers * 2` (read + write per layer)

### Latency Metrics

- **TTFT**: Time-To-First-Token
  - From client-side measurements (if `--latency-file` provided)
  - Or from Prometheus histogram `sglang:time_to_first_token_seconds` (if available)
  
- **TBT**: Time-Between-Tokens
  - From client-side measurements (if `--latency-file` provided)
  - Or from Prometheus histogram `sglang:inter_token_latency_seconds` (if available)

### GPU Utilization

- Sampled using `nvidia-smi --query-gpu=utilization.gpu`
- Average across all GPUs at each sampling interval

### Queue Wait Time

- Collected from Prometheus metric `sglang:avg_request_queue_latency`
- Averaged across all workers

## Workflow Example

### Complete Workflow

1. **Start AFD workers**:
   ```bash
   # Attn worker
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export AFD_SCHED_HOST=<ffn_ip>
   python -m sglang.launch_server \
       --model-path <model> \
       --afd-perspective attn \
       --afd-mirco-batch 3 \
       --port 30000
   
   # FFN worker
   export CUDA_VISIBLE_DEVICES=4,5,6,7
   export AFD_SCHED_HOST=<ffn_ip>
   python -m sglang.launch_server \
       --model-path <model> \
       --skip-server-warmup \
       --afd-perspective ffn \
       --afd-mirco-batch 3 \
       --port 30001
   ```

2. **Run benchmark with real workload** (in another terminal):
   ```bash
   python -m sglang.bench_serving \
       --backend sglang \
       --port 30000 \
       --dataset-name random \
       --num-prompts 1000 \
       --random-input-len 512 \
       --random-output-len 256
   ```

3. **Collect metrics simultaneously**:
   ```bash
   python benchmark/afd/collect_afd_metrics.py \
       --attn-metrics-url http://localhost:30000/metrics \
       --ffn-metrics-url http://localhost:30001/metrics \
       --duration 600 \
       --interval 1.0 \
       --output afd_metrics.json \
       --plot afd_metrics.png
   ```

4. **Analyze results**:
   - Open `afd_metrics.png` to visualize metric correlations
   - Check correlation coefficients in the plot
   - The metric with highest correlation with workload is the best indicator

## Notes

1. **Time Alignment**: All metrics are collected at the same time intervals with synchronized timestamps, ensuring proper correlation analysis.

2. **Workload Definition**: Workload is defined as KV cache-missed prefill input, representing the actual computational load on GPUs.

3. **KV Operations Estimation**: The current implementation provides a simplified estimate. For accurate measurement, you may need to instrument the attention layer code to count actual KV cache operations.

4. **Latency Data**: If you have real request traces, use `extract_latency_from_requests.py` to extract TTFT/TBT and include them in the analysis.

5. **GPU Monitoring**: Requires `nvidia-smi` to be available in PATH.

## Example Workflow

1. **Start AFD Attn worker**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export AFD_SCHED_HOST=<ffn_ip>
   python -m sglang.launch_server \
       --model-path <model> \
       --afd-perspective attn \
       --afd-mirco-batch 3
   ```

2. **Start AFD FFN worker**:
   ```bash
   export CUDA_VISIBLE_DEVICES=4,5,6,7
   export AFD_SCHED_HOST=<ffn_ip>
   python -m sglang.launch_server \
       --model-path <model> \
       --skip-server-warmup \
       --afd-perspective ffn \
       --afd-mirco-batch 3 \
       --port 30001
   ```

3. **Run benchmark for Attn worker**:
   ```bash
   python benchmark/afd/bench_afd_metrics.py \
       --api-url http://localhost:30000 \
       --afd-perspective attn \
       --duration 600 \
       --output attn_metrics.json
   ```

4. **Run benchmark for FFN worker**:
   ```bash
   python benchmark/afd/bench_afd_metrics.py \
       --api-url http://localhost:30001 \
       --afd-perspective ffn \
       --duration 600 \
       --output ffn_metrics.json
   ```

5. **Compare results**: Analyze the JSON outputs to compare metrics between attn and ffn workers, and evaluate autoscaling decisions.
