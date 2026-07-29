# HexGen-3 Paper-Mimic Experiment Results

These are simulator estimates, not a reproduction of the paper's private run.

## Static WildGPT

| Baseline | Arch | Capacity | Throughput req/s | Req/$ | P99 s | P99 cost $ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SGLang_Homo_PD | PD | {'NVDA:H100:SXM': 16} | 328.4655 | 24073.20 | 0.0322 | 0.000439 |
| MegaScaleInfer_Homo_AFD | AFD | {'NVDA:H100:SXM': 16} | 222.6106 | 16315.11 | 0.0369 | 0.000504 |
| HexGen2_Hetero_PD | PD | {'NVDA:H100:SXM': 8, 'NVDA:H20': 16} | 1272.0627 | 84553.65 | 0.0109 | 0.000165 |
| HexGen3_Homo_AFD | AFD | {'NVDA:H100:SXM': 16} | 468.9439 | 34368.85 | 0.0221 | 0.000302 |
| HexGen3_Hetero_AFD | AFD | {'NVDA:H100:SXM': 8, 'NVDA:H20': 16} | 1120.3072 | 74466.51 | 0.0163 | 0.000245 |
| HexGen3_Homo_UniformAllocation | AFD | {'NVDA:H100:SXM': 16} | 327.4700 | 24000.24 | 0.0302 | 0.000412 |
| HexGen3_Homo_UniformParallelism | AFD | {'NVDA:H100:SXM': 16} | 400.6991 | 29367.20 | 0.0277 | 0.000378 |