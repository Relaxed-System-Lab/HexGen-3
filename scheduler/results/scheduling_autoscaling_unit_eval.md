# Scheduling And Autoscaling Unit Evaluation

These results use the checked-in scheduling framework with:

- `match_attn_ffn_dp=True`
- autoscaling decode worker GPU choices `{1, 2, 4, 8}`
- all-H100 clusters
- 30 maximum global scheduling iterations

## Unit Tests

```bash
python3 -m unittest tests.test_hexgen3_scheduler tests.test_scheduling_autoscaling_evaluation -v
python3 -m compileall simulator/scheduling tests/test_hexgen3_scheduler.py tests/test_scheduling_autoscaling_evaluation.py
```

Status: passed.

## Large Cluster Scheduling

Workload: synthetic mixed prompt/decode workload, 4 req/s, 30% long requests.

| Cluster | Time s | Iter | Throughput req/s | Latency s | Cost/hr | Req/$ | Allocation | Attn DP | FFN DP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 24xH100 | 0.125 | 27 | 106.6952 | 0.0280 | 37.44 | 10259.16 | pre=3, attn=8, ffn=13 | 7 | 7 |
| 32xH100 | 0.197 | 21 | 142.2603 | 0.0207 | 49.92 | 10259.16 | pre=4, attn=11, ffn=17 | 9 | 9 |
| 48xH100 | 0.858 | 30 | 218.2281 | 0.0133 | 74.88 | 10491.73 | pre=7, attn=15, ffn=26 | 8 | 8 |
| 64xH100 | 2.322 | 30 | 260.2933 | 0.0107 | 99.84 | 9385.57 | pre=9, attn=20, ffn=35 | 20 | 20 |
| 128xH100 | 8.806 | 30 | 371.2955 | 0.0056 | 199.68 | 6694.03 | pre=26, attn=41, ffn=61 | 41 | 41 |

## 16-GPU Autoscaling With Parallelism

| Window | Arrival req/s | Throughput req/s | Latency s | P99 s | Allocation | Pre | Attn | FFN | Attn DP | FFN DP |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| low-5m | 2.0 | 58.4832 | 0.0463 | 0.1372 | pre=2, attn=4, ffn=8 | dp=2,tp=1 | dp=2,tp=2 | dp=2,tp=4 | 2 | 2 |
| surge-10m | 80.0 | 26.2162 | 6.3657 | 13.1259 | pre=3, attn=8, ffn=4 | dp=3,tp=1 | dp=2,tp=4 | dp=2,tp=2 | 2 | 2 |
| peak-15m | 110.0 | 32.3091 | 5.3656 | 11.1476 | pre=4, attn=4, ffn=8 | dp=4,tp=1 | dp=2,tp=2 | dp=2,tp=4 | 2 | 2 |
| recovery-30m | 5.0 | 29.2416 | 0.0876 | 0.3062 | pre=2, attn=2, ffn=4 | dp=2,tp=1 | dp=1,tp=2 | dp=1,tp=4 | 1 | 1 |

The autoscaling allocation constraint is visible in every window: attention and
FFN use only `{1, 2, 4, 8}` GPUs. The local scheduler also keeps total attention
DP equal to total FFN DP while allowing different TP values.
