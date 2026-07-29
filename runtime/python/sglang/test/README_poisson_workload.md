# Poisson Process Workload Testing

This test script implements workload generation using Poisson processes for evaluating baseline frameworks with Qwen-MoE models and real-world chatbot workloads from the ShareGPT dataset.

## Features

1. **Standard (Homogeneous) Poisson Process**: Constant arrival rate over time
2. **Non-Homogeneous Poisson Process**: Varying arrival rates over time to simulate real-world workload fluctuations

Following prior work (Li et al., 2023; Miao et al., 2024), we generate inference workloads using a Poisson process with varying request arrival rates over time.

## Prerequisites

1. Start the SGLang server with a Qwen-MoE model:
   ```bash
   python -m sglang.launch_server \
       --model-path Qwen/Qwen2.5-MoE-A14B-Chat \
       --port 30000
   ```

2. The script will automatically download the ShareGPT dataset if not provided.

## Usage

### Fake Mode (Visualization Only)

Generate and visualize Poisson process without sending actual requests:

```bash
python -m sglang.test.test_poisson_workload \
    --num-requests 1000 \
    --request-rate 10.0 \
    --poisson-type homogeneous \
    --fake \
    --plot-output poisson_homogeneous.png
```

For non-homogeneous Poisson:
```bash
python -m sglang.test.test_poisson_workload \
    --num-requests 1000 \
    --base-request-rate 10.0 \
    --poisson-type non-homogeneous \
    --rate-variation sin \
    --variation-amplitude 0.5 \
    --variation-period 60.0 \
    --fake \
    --plot-output poisson_nonhomogeneous.png
```

**Note**: In fake mode, `--model-path` is not required. The script will generate arrival times and create visualization plots without connecting to any server.

### Standard (Homogeneous) Poisson Process

Generate requests with a constant arrival rate:

```bash
python -m sglang.test.test_poisson_workload \
    --model-path Qwen/Qwen2.5-MoE-A14B-Chat \
    --num-requests 1000 \
    --request-rate 10.0 \
    --poisson-type homogeneous \
    --output-file results_homogeneous.json
```

### Non-Homogeneous Poisson Process

Generate requests with varying arrival rates:

#### Sinusoidal Variation
```bash
python -m sglang.test.test_poisson_workload \
    --model-path Qwen/Qwen2.5-MoE-A14B-Chat \
    --num-requests 1000 \
    --base-request-rate 10.0 \
    --poisson-type non-homogeneous \
    --rate-variation sin \
    --variation-amplitude 0.5 \
    --variation-period 60.0 \
    --output-file results_nonhomogeneous_sin.json
```

#### Linear Variation
```bash
python -m sglang.test.test_poisson_workload \
    --model-path Qwen/Qwen2.5-MoE-A14B-Chat \
    --num-requests 1000 \
    --base-request-rate 10.0 \
    --poisson-type non-homogeneous \
    --rate-variation linear \
    --variation-amplitude 0.5 \
    --variation-period 60.0 \
    --output-file results_nonhomogeneous_linear.json
```

#### Step Variation
```bash
python -m sglang.test.test_poisson_workload \
    --model-path Qwen/Qwen2.5-MoE-A14B-Chat \
    --num-requests 1000 \
    --base-request-rate 10.0 \
    --poisson-type non-homogeneous \
    --rate-variation step \
    --variation-amplitude 0.5 \
    --variation-period 60.0 \
    --output-file results_nonhomogeneous_step.json
```

## Arguments

### Server Configuration
- `--backend`: Backend to use (default: "sglang")
- `--host`: Server host (default: "localhost")
- `--port`: Server port (default: 30000)
- `--base-url`: Server base URL (alternative to host/port)

### Model Configuration
- `--model-path`: Path to the model (required)
- `--tokenizer`: Tokenizer path (defaults to model-path)

### Dataset Configuration
- `--dataset-path`: Path to ShareGPT dataset JSON file (will download if not provided)
- `--num-requests`: Number of requests to generate (default: 1000)
- `--sharegpt-output-len`: Fixed output length for ShareGPT requests
- `--sharegpt-context-len`: Maximum context length for ShareGPT requests

### Poisson Process Configuration
- `--poisson-type`: Type of Poisson process - "homogeneous" or "non-homogeneous" (default: "homogeneous")
- `--request-rate`: Request arrival rate for homogeneous Poisson (requests per second)
- `--base-request-rate`: Base request arrival rate (default: 10.0)
- `--rate-variation`: Type of rate variation for non-homogeneous Poisson - "sin", "linear", or "step" (default: "sin")
- `--variation-amplitude`: Amplitude of rate variation 0-1 (default: 0.5)
- `--variation-period`: Period of rate variation in seconds (default: 60.0)

### Benchmark Configuration
- `--max-concurrency`: Maximum number of concurrent requests
- `--seed`: Random seed (default: 42)
- `--disable-tqdm`: Disable progress bar
- `--output-file`: Output file for results (JSON format)
- `--fake`: Fake mode - generate Poisson process but don't send actual requests. Instead, plot request count over time.
- `--plot-output`: Output file path for the plot (only used in fake mode). If not specified, generates a default filename.

## Output

### Normal Mode
The script outputs:
- Request throughput (req/s)
- Input/Output/Total token throughput (tok/s)
- End-to-end latency statistics (mean, median, P99)
- Time to first token (TTFT) statistics
- Inter-token latency (ITL) statistics

Results are also saved to a JSON file if `--output-file` is specified.

### Fake Mode
In fake mode, the script:
- Generates Poisson process arrival times
- Displays statistics about the generated process (total duration, average inter-arrival time, etc.)
- Creates visualization plots:
  - **Top plot**: Cumulative request count over time
  - **Bottom plot**: 
    - For homogeneous Poisson: Histogram of inter-arrival times with theoretical exponential distribution overlay
    - For non-homogeneous Poisson: Request rate over time (theoretical rate function and estimated instantaneous rate)

The plot is saved to the specified `--plot-output` file, or a default filename if not specified.

## Implementation Details

### Homogeneous Poisson Process
Uses exponential distribution for inter-arrival times:
- Inter-arrival time ~ Exponential(λ), where λ = request_rate

### Non-Homogeneous Poisson Process
Uses the thinning algorithm (Lewis-Shedler method) to generate events from a non-homogeneous Poisson process:
1. Generate candidate events from a homogeneous Poisson process with maximum rate
2. Accept each event with probability λ(t) / λ_max, where λ(t) is the time-varying rate function

## References

- Li, et al. (2023) - Prior work on workload generation
- Miao, et al. (2024) - Prior work on workload generation
- ShareGPT dataset: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
