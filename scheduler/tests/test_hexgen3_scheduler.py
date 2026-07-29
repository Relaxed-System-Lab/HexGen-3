import unittest

from simulator.scheduling import (
    AllocationMatrix,
    AutoscalingConfig,
    GlobalScheduler,
    GlobalSchedulerConfig,
    HexGenSchedulingFramework,
    LocalScheduler,
    LocalSchedulerConfig,
    ParallelismReplica,
    ParallelismStrategy,
    ThroughputProfile,
    WorkloadProfile,
)


class FakeEstimator:
    def estimate_slice_throughput(self, worker_type, hardware, strategy, workload):
        worker_base = {"pre": 5.0, "attn": 2.0, "ffn": 1.0}[worker_type]
        hw_mult = {"fast": 2.0, "cheap": 1.0}[hardware]
        return worker_base * hw_mult * strategy.gpus * (1.0 + 0.05 * strategy.tp)

    def estimate_latency(self, workload, throughput):
        return 1.0 / max(throughput.bottleneck, 1e-9)

    def estimate_tail_latency(self, workload, throughput):
        latency = self.estimate_latency(workload, throughput)
        return {"mean": latency, "p50": latency, "p95": latency, "p99": latency}

    def estimate_cost_per_hour(self, allocation):
        return allocation.total_for_hardware("fast") * 2.0 + allocation.total_for_hardware("cheap")


class TestHexGen3Scheduler(unittest.TestCase):
    def test_local_scheduler_returns_valid_parallelism_for_each_slice(self):
        workload = WorkloadProfile.synthetic(arrival_rate=2.0, samples=8)
        allocation = AllocationMatrix.uniform({"fast": 3, "cheap": 3})
        scheduler = LocalScheduler(FakeEstimator(), LocalSchedulerConfig(node_gpus=8))

        plan = scheduler.evaluate(workload, allocation)

        self.assertGreater(plan.throughput.bottleneck, 0)
        for (worker, hardware), strategy in plan.parallelism.items():
            self.assertIsInstance(strategy, ParallelismStrategy)
            self.assertEqual(strategy.gpus, allocation.get(worker, hardware))

    def test_local_scheduler_enumerates_non_uniform_tp_partitions(self):
        scheduler = LocalScheduler(FakeEstimator(), LocalSchedulerConfig(node_gpus=8))

        strategies = scheduler.enumerate_strategies(3)
        replica_layouts = {strategy.as_tuple() for strategy in strategies}

        self.assertIn(((1, 1), (2, 1)), replica_layouts)
        self.assertIn(((1, 1), (1, 1), (1, 1)), replica_layouts)
        self.assertIn(((3, 1),), replica_layouts)

    def test_expert_parallel_constraints_include_cross_replica_ep(self):
        scheduler = LocalScheduler(
            FakeEstimator(),
            LocalSchedulerConfig(node_gpus=8, enable_expert_parallel=True, num_experts=8),
        )

        strategies = scheduler.enumerate_strategies(4)
        replica_layouts = {strategy.as_tuple() for strategy in strategies}

        self.assertIn(((2, 4), (2, 4)), replica_layouts)
        self.assertIn(((4, 4),), replica_layouts)

    def test_warm_start_projection_preserves_tp_ep_when_scaling_dp(self):
        scheduler = LocalScheduler(FakeEstimator(), LocalSchedulerConfig(node_gpus=8))
        previous = ParallelismStrategy(dp=2, tp=2, ep=1)

        projected = scheduler.project_strategy(previous, 6)

        self.assertIsNotNone(projected)
        self.assertEqual(projected.as_tuple(), ((2, 1), (2, 1), (2, 1)))

    def test_local_scheduler_matches_attention_and_ffn_total_dp(self):
        workload = WorkloadProfile.synthetic(arrival_rate=2.0, samples=8)
        allocation = AllocationMatrix.zeros({"fast"})
        allocation.set("pre", "fast", 1)
        allocation.set("attn", "fast", 4)
        allocation.set("ffn", "fast", 8)
        scheduler = LocalScheduler(
            FakeEstimator(),
            LocalSchedulerConfig(node_gpus=8, match_attn_ffn_dp=True),
        )

        plan = scheduler.evaluate(workload, allocation)

        attn_dp = sum(
            strategy.dp
            for (worker, _), strategy in plan.parallelism.items()
            if worker == "attn"
        )
        ffn_dp = sum(
            strategy.dp
            for (worker, _), strategy in plan.parallelism.items()
            if worker == "ffn"
        )
        self.assertEqual(attn_dp, ffn_dp)
        self.assertNotEqual(
            plan.parallelism[("attn", "fast")].gpus,
            plan.parallelism[("ffn", "fast")].gpus,
        )

    def test_global_scheduler_keeps_capacity_constraints(self):
        workload = WorkloadProfile.synthetic(arrival_rate=2.0, samples=8)
        capacity = {"fast": 4, "cheap": 4}
        local = LocalScheduler(FakeEstimator(), LocalSchedulerConfig(node_gpus=8))
        global_scheduler = GlobalScheduler(
            local,
            GlobalSchedulerConfig(iterations=10, stability_iterations=5, seed=3),
        )

        plan = global_scheduler.optimize(workload, capacity)

        plan.allocation.validate(capacity)
        self.assertGreater(plan.throughput.bottleneck, 0)
        self.assertEqual(plan.allocation.total_for_hardware("fast"), capacity["fast"])
        self.assertEqual(plan.allocation.total_for_hardware("cheap"), capacity["cheap"])
        self.assertGreaterEqual(plan.iterations, 1)

    def test_global_scheduler_can_allocate_from_empty_source(self):
        workload = WorkloadProfile.synthetic(arrival_rate=2.0, samples=8)
        capacity = {"fast": 5}
        initial = AllocationMatrix.zeros(capacity.keys())
        initial.set("pre", "fast", 1)
        initial.set("attn", "fast", 1)
        initial.set("ffn", "fast", 1)
        local = LocalScheduler(FakeEstimator(), LocalSchedulerConfig(node_gpus=8))
        global_scheduler = GlobalScheduler(
            local,
            GlobalSchedulerConfig(iterations=8, stability_iterations=8, seed=1, allow_empty_source=True),
        )

        plan = global_scheduler.optimize(workload, capacity, initial_allocation=initial)

        plan.allocation.validate(capacity)
        self.assertGreater(plan.allocation.total_for_hardware("fast"), initial.total_for_hardware("fast"))

    def test_framework_reschedule_scales_allocation_with_warm_start(self):
        workload = WorkloadProfile.synthetic(arrival_rate=2.0, samples=8)
        capacity = {"fast": 6}
        allocation = AllocationMatrix.zeros(capacity.keys())
        allocation.set("pre", "fast", 1)
        allocation.set("attn", "fast", 1)
        allocation.set("ffn", "fast", 1)
        previous_strategy = ParallelismStrategy.from_replicas((ParallelismReplica(tp=1, ep=1),))
        previous_plan = type(
            "Plan",
            (),
            {
                "allocation": allocation,
                "parallelism": {
                    ("pre", "fast"): previous_strategy,
                    ("attn", "fast"): previous_strategy,
                    ("ffn", "fast"): previous_strategy,
                },
                "throughput": ThroughputProfile({"pre": 10.0, "attn": 10.0, "ffn": 0.5}),
            },
        )()
        framework = HexGenSchedulingFramework(
            local_config=LocalSchedulerConfig(node_gpus=8),
            global_config=GlobalSchedulerConfig(iterations=4, stability_iterations=4, seed=2),
            autoscaling_config=AutoscalingConfig(target_utilization=0.75, max_scale_factor=2.0),
        )
        framework.estimator = FakeEstimator()
        framework.local_scheduler.estimator = framework.estimator

        scaled = framework.proportional_scale_allocation(workload, previous_plan, capacity)

        self.assertGreaterEqual(scaled.get("ffn", "fast"), 2)

    def test_framework_reschedule_quantizes_decode_worker_allocations(self):
        workload = WorkloadProfile.synthetic(arrival_rate=2.0, samples=8)
        surge_workload = WorkloadProfile.synthetic(arrival_rate=20.0, samples=8)
        capacity = {"fast": 16}
        framework = HexGenSchedulingFramework(
            local_config=LocalSchedulerConfig(node_gpus=8, match_attn_ffn_dp=True),
            global_config=GlobalSchedulerConfig(iterations=6, stability_iterations=6, seed=5),
            autoscaling_config=AutoscalingConfig(
                target_utilization=0.75,
                max_scale_factor=2.0,
                decode_worker_gpu_choices=(1, 2, 4, 8),
            ),
        )
        framework.estimator = FakeEstimator()
        framework.local_scheduler.estimator = framework.estimator
        previous_plan = framework.optimize(workload, capacity)

        scaled = framework.proportional_scale_allocation(surge_workload, previous_plan, capacity)
        plan = framework.reschedule(surge_workload, previous_plan, capacity)

        for allocation in (scaled, plan.allocation):
            self.assertIn(sum(allocation.values["attn"].values()), {1, 2, 4, 8})
            self.assertIn(sum(allocation.values["ffn"].values()), {1, 2, 4, 8})

        attn_dp = sum(
            strategy.dp
            for (worker, _), strategy in plan.parallelism.items()
            if worker == "attn"
        )
        ffn_dp = sum(
            strategy.dp
            for (worker, _), strategy in plan.parallelism.items()
            if worker == "ffn"
        )
        self.assertEqual(attn_dp, ffn_dp)


if __name__ == "__main__":
    unittest.main()
