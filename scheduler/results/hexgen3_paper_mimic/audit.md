# Paper-Mimic Audit

This audit compares the local framework against the HexGen-3 paper sections
used for implementation and experiments.

## Covered

- Static scheduler objective: maximize bottleneck throughput across `pre`,
  `attn`, and `ffn` workers under GPU capacity constraints.
- Global scheduler: guided simulated annealing with destination probability
  weighted by reciprocal worker throughput and source probability weighted by
  worker throughput.
- Global action source `empty`: supported for newly available GPUs during
  rescheduling.
- Local scheduler: constrained enumeration for non-zero `(worker, hardware)`
  allocation cells.
- Local constraints: non-uniform TP partitions satisfying
  `sum(replica.tp) == allocated_gpus`, EP divisibility, cross-replica EP
  identical-TP rule, and node-local `max(TP, EP) <= Gnode`.
- Autoscaling rescheduling: proportional allocation projection, warm-start
  strategy projection, and reload-cost-aware local search.
- Experiment configs: paper homogeneous and heterogeneous hardware/cost setup,
  Qwen3-30B-A3B model id, WildGPT/WildChat-derived trace, and Table-1 dynamic
  load/type sequence.
- AFD simulator integration: per-instance attention and FFN analyzers for
  heterogeneous hardware instead of collapsing all AFD timing to the first
  attention node.

## Intentional Approximations

- The public WildGPT trace used here is a WildChat-derived sample, not the
  paper authors' private subsample.
- Only WildGPT is materialized locally. OpenThoughts, OpenR1-Math, and
  NuminaMath dynamic workload types are mimicked with documented scaling factors
  over the WildGPT token distribution.
- SGLang, MegaScale-Infer, HexGen-2, and HeteroScale are approximated with the
  same local roofline estimator rather than running those serving systems.
- The simulator is analytical. It estimates throughput and queueing tails; it is
  not a cycle-accurate distributed runtime.
- Qwen3-30B-A3B uses a local offline config fallback when Hugging Face config
  downloads fail. The active MoE FFN size is represented, while full MoE runtime
  routing is approximated through EP scaling.

No remaining paper detail can be implemented faithfully without either the
authors' private trace/configuration choices or full implementations of the
external baseline systems.
