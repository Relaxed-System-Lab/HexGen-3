# Autoscaling Run Log

## Full Output

```
Actual arrivals by window:
  Window 0: arrivals=106, avg_input=22.7, avg_output=45.2
  Window 1: arrivals=163, avg_input=21.5, avg_output=149.7
  Window 2: arrivals=16, avg_input=16.9, avg_output=11.6

============================================================
BASELINE (P-D fixed H100 x4 decode)
============================================================
{
  "completed": 285,
  "avg_latency": 1.484961128829732,
  "p50_latency": 1.4785919318860117,
  "p95_latency": 2.5425835074557424,
  "p99_latency": 2.7212086153108217,
  "sim_time": 60.608378734663006
}

------------------------------------------------------------
Baseline per-window (fixed config)
Event loop statistics:
  Events processed: 14603
  Simulation time: 22.070s
Window 0: P95=2.412s
Event loop statistics:
  Events processed: 17318
  Simulation time: 22.005s
Window 1: P95=2.597s
Event loop statistics:
  Events processed: 4113
  Simulation time: 20.366s
Window 2: P95=2.148s

------------------------------------------------------------
AFD baseline (no autoscaling)
{
  "completed": 285,
  "avg_latency": 2.2616591260696683,
  "p95_latency": 2.117101203125916,
  "sim_time": 240.93772968706853
}

------------------------------------------------------------
AFD Baseline per-window (fixed config)
Event loop statistics:
  Events processed: 733
  Simulation time: 21.615s
Window 0: P95=1.894s
Event loop statistics:
  Events processed: 1005
  Simulation time: 20.948s
Window 1: P95=1.900s
Event loop statistics:
  Events processed: 115
  Simulation time: 20.000s
Window 2: P95=1.394s

------------------------------------------------------------
P-D Autoscaling per-window decisions
  Window 0 (reference): P95=2.501s
Event loop statistics:
  Events processed: 26318
  Simulation time: 21.819s
  window_1 (decode_repl=8): P95=2.456s, decision=scale_up, cost_delta=$0.10
Event loop statistics:
  Events processed: 3444
  Simulation time: 20.334s
  window_2 (decode_repl=2): P95=2.426s, decision=scale_down, cost_delta=$-0.04

============================================================
P-D Autoscaling
============================================================
  Completed: 285
  Avg latency: 1.433s
  P95 latency: 2.487s

------------------------------------------------------------
AFD Autoscaling per-window decisions
  Window 0 (reference): P95=1.809s
Event loop statistics:
  Events processed: 1071
  Simulation time: 20.743s
  window_1: P95=1.472s, decision=attn:scale_up, ffn:scale_up, cost_delta=$0.06
Event loop statistics:
  Events processed: 115
  Simulation time: 20.000s
  window_2: P95=1.400s, decision=attn:scale_down, ffn:scale_down, cost_delta=$-0.01

============================================================
AFD Autoscaling
============================================================
  Completed: 285
  Avg latency: 0.999s
  P95 latency: 1.643s

============================================================
FINAL COMPARISON (P95 ONLY)
============================================================
P-D Baseline: P95 2.543s
P-D Autoscaling: P95 2.487s
AFD Baseline: P95 2.117s
AFD Autoscaling: P95 1.643s
```

## Final P95 Latencies

- P-D Baseline: P95 2.543s
- P-D Autoscaling: P95 2.487s
- AFD Baseline: P95 2.117s
- AFD Autoscaling: P95 1.643s
