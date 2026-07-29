# HexGen-3 Paper-Mimic Experiment Results

These are simulator estimates, not a reproduction of the paper's private run.

## Static WildGPT

| Baseline | Arch | Capacity | Throughput req/s | Req/$ | P99 s | P99 cost $ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SGLang_Homo_PD | PD | {'NVDA:H100:SXM': 16} | 104.4328 | 7653.87 | 0.1146 | 0.001564 |
| MegaScaleInfer_Homo_AFD | AFD | {'NVDA:H100:SXM': 16} | 60.3354 | 4421.97 | 0.1213 | 0.001655 |
| HexGen2_Hetero_PD | PD | {'NVDA:H100:SXM': 8, 'NVDA:H20': 16} | 486.3313 | 32326.30 | 0.0266 | 0.000400 |
| HexGen3_Homo_AFD | AFD | {'NVDA:H100:SXM': 16} | 610.4178 | 44737.46 | 0.0213 | 0.000291 |
| HexGen3_Hetero_AFD | AFD | {'NVDA:H100:SXM': 8, 'NVDA:H20': 16} | 936.8091 | 62269.44 | 0.0237 | 0.000356 |
| HexGen3_Homo_UniformAllocation | AFD | {'NVDA:H100:SXM': 16} | 327.4700 | 24000.24 | 0.0302 | 0.000412 |
| HexGen3_Homo_UniformParallelism | AFD | {'NVDA:H100:SXM': 16} | 424.4218 | 31105.83 | 0.0251 | 0.000342 |

## Dynamic Table-1 Mimic

Hour 1: load=6600, type=3, resources=12 GPUs

| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |
| --- | ---: | ---: | ---: | --- |
| SGLang_Autoscale_PD | 44.0677 | 4306.29 | 0.2711 | {'NVDA:H100:SXM': 12} |
| HeteroScale_Autoscale_PD | 72.2223 | 9601.19 | 0.1627 | {'NVDA:H100:SXM': 4, 'NVDA:H20': 8} |
| HexGen3_Autoscale_AFD | 154.2519 | 20506.16 | 0.1308 | {'NVDA:H100:SXM': 4, 'NVDA:H20': 8} |

Hour 2: load=11200, type=3, resources=18 GPUs

| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |
| --- | ---: | ---: | ---: | --- |
| SGLang_Autoscale_PD | 34.1688 | 2504.23 | 0.3750 | {'NVDA:H100:SXM': 16} |
| HeteroScale_Autoscale_PD | 112.2667 | 9949.78 | 0.1070 | {'NVDA:H100:SXM': 6, 'NVDA:H20': 12} |
| HexGen3_Autoscale_AFD | 269.5262 | 23887.11 | 0.0462 | {'NVDA:H100:SXM': 6, 'NVDA:H20': 12} |

Hour 3: load=61400, type=1, resources=18 GPUs

| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |
| --- | ---: | ---: | ---: | --- |
| SGLang_Autoscale_PD | 104.4328 | 7653.87 | 0.1524 | {'NVDA:H100:SXM': 16} |
| HeteroScale_Autoscale_PD | 358.3233 | 31756.87 | 0.0422 | {'NVDA:H100:SXM': 6, 'NVDA:H20': 12} |
| HexGen3_Autoscale_AFD | 875.4665 | 77589.35 | 0.0214 | {'NVDA:H100:SXM': 6, 'NVDA:H20': 12} |

Hour 4: load=129000, type=1, resources=24 GPUs

| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |
| --- | ---: | ---: | ---: | --- |
| SGLang_Autoscale_PD | 104.4328 | 7653.87 | 0.1819 | {'NVDA:H100:SXM': 16} |
| HeteroScale_Autoscale_PD | 486.3313 | 32326.30 | 0.0400 | {'NVDA:H100:SXM': 8, 'NVDA:H20': 16} |
| HexGen3_Autoscale_AFD | 936.8091 | 62269.44 | 0.0273 | {'NVDA:H100:SXM': 8, 'NVDA:H20': 16} |

Hour 5: load=13900, type=4, resources=14 GPUs

| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |
| --- | ---: | ---: | ---: | --- |
| SGLang_Autoscale_PD | 59.8093 | 5009.62 | 0.2099 | {'NVDA:H100:SXM': 14} |
| HeteroScale_Autoscale_PD | 91.4688 | 10290.24 | 0.1346 | {'NVDA:H100:SXM': 5, 'NVDA:H20': 9} |
| HexGen3_Autoscale_AFD | 185.5619 | 20875.72 | 0.1157 | {'NVDA:H100:SXM': 5, 'NVDA:H20': 9} |

Hour 6: load=8200, type=2, resources=12 GPUs

| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |
| --- | ---: | ---: | ---: | --- |
| SGLang_Autoscale_PD | 56.5890 | 5529.87 | 0.2170 | {'NVDA:H100:SXM': 12} |
| HeteroScale_Autoscale_PD | 92.6206 | 12312.94 | 0.1281 | {'NVDA:H100:SXM': 4, 'NVDA:H20': 8} |
| HexGen3_Autoscale_AFD | 208.3066 | 27692.16 | 0.0800 | {'NVDA:H100:SXM': 4, 'NVDA:H20': 8} |
