# HexGen-3 Scheduling Framework

Simulator-backed implementation of the HexGen-3 scheduling framework for
Attention-FFN Disaggregated serving. The repo includes the copied LLM serving
simulator, replica-aware local scheduling, guided simulated-annealing global
scheduling, autoscaling/rescheduling helpers, a compact WildGPT/WildChat trace,
and paper-mimic experiment outputs.

## Quick Start

```bash
python3 -m unittest tests.test_hexgen3_scheduler
python3 cli/run_hexgen3_scheduler.py \
  --capacity '{"NVDA:H100:SXM": 4, "NVDA:H20": 4}' \
  --arrival-rate 4 \
  --cost-aware \
  --output output_test/hexgen3_plan.json
```

Run the paper-mimic WildGPT experiment:

```bash
python3 cli/run_hexgen3_paper_experiments.py \
  --trace traces/wildgpt_trace.jsonl \
  --samples 512 \
  --iterations 18 \
  --stability-iterations 8 \
  --output results/hexgen3_paper_mimic/results.json \
  --summary-md results/hexgen3_paper_mimic/summary.md
```

## Included Results

Static WildGPT simulator estimates:

| Baseline | Throughput req/s | Req/$ | P99 s |
|---|---:|---:|---:|
| SGLang Homo PD | 104.43 | 7653.87 | 0.1146 |
| MegaScale-Infer Homo AFD | 60.34 | 4421.97 | 0.1213 |
| HexGen-2 Hetero PD | 486.33 | 32326.30 | 0.0266 |
| HexGen-3 Homo AFD | 610.42 | 44737.46 | 0.0213 |
| HexGen-3 Hetero AFD | 936.81 | 62269.44 | 0.0237 |

Dynamic Table-1 mimic averages:

| Baseline | Avg throughput req/s | Avg req/$ | Avg P99 s |
|---|---:|---:|---:|
| SGLang Autoscale PD | 67.25 | 5442.96 | 0.2345 |
| HeteroScale Autoscale PD | 202.21 | 17706.22 | 0.1024 |
| HexGen-3 Autoscale AFD | 438.32 | 38803.32 | 0.0702 |

Full outputs are in `results/hexgen3_paper_mimic/`. This is an analytical
simulator estimate, not a reproduction of the paper authors' private trace or
external serving-system implementations.
