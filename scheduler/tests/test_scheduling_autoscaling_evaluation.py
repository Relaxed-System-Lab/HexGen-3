import unittest

from simulator.scheduling import (
    AutoscalingConfig,
    GlobalSchedulerConfig,
    HexGenSchedulingFramework,
    LoadWindow,
    LocalSchedulerConfig,
    SchedulingCase,
    WorkloadProfile,
    measure_scheduling_cases,
    simulate_autoscaling_windows,
)


class EvalFakeEstimator:
    def estimate_slice_throughput(self, worker_type, hardware, strategy, workload):
        worker_base = {"pre": 5.0, "attn": 2.0, "ffn": 1.0}[worker_type]
        hardware_mult = {"fast": 2.0, "cheap": 1.0}[hardware]
        token_penalty = 512.0 / max(workload.mean_input + workload.mean_output, 1)
        return worker_base * hardware_mult * strategy.gpus * (1.0 + 0.05 * strategy.tp) * token_penalty

    def estimate_latency(self, workload, throughput):
        bottleneck = max(throughput.bottleneck, 1e-9)
        utilization = min(0.99, workload.arrival_rate / bottleneck)
        return (1.0 / bottleneck) / max(1.0 - utilization, 1e-9)

    def estimate_tail_latency(self, workload, throughput):
        mean = self.estimate_latency(workload, throughput)
        return {"mean": mean, "p50": mean, "p95": mean * 1.5, "p99": mean * 2.0}

    def estimate_cost_per_hour(self, allocation):
        return allocation.total_for_hardware("fast") * 2.0 + allocation.total_for_hardware("cheap")


def make_test_framework(iterations=8):
    framework = HexGenSchedulingFramework(
        local_config=LocalSchedulerConfig(
            node_gpus=8,
            max_local_strategies=128,
            cost_aware=True,
            model_size_gb=8.0,
        ),
        global_config=GlobalSchedulerConfig(
            iterations=iterations,
            stability_iterations=iterations,
            seed=11,
            keep_history=False,
        ),
        autoscaling_config=AutoscalingConfig(
            target_utilization=0.65,
            hysteresis=0.05,
            min_scale_factor=0.5,
            max_scale_factor=2.0,
        ),
    )
    framework.estimator = EvalFakeEstimator()
    framework.local_scheduler.estimator = framework.estimator
    return framework


class TestSchedulingAutoscalingEvaluation(unittest.TestCase):
    def test_measures_scheduling_time_and_outcomes_for_cluster_sizes(self):
        workload = WorkloadProfile.synthetic(
            arrival_rate=1.5,
            short_input=256,
            long_input=768,
            short_output=64,
            long_output=192,
            samples=12,
            max_batch_size=4,
        )
        cases = [
            SchedulingCase("small-fast", {"fast": 3}),
            SchedulingCase("medium-fast", {"fast": 6}),
            SchedulingCase("heterogeneous", {"fast": 4, "cheap": 4}),
        ]

        measurements = measure_scheduling_cases(
            lambda: make_test_framework(iterations=6),
            workload,
            cases,
        )

        self.assertEqual(len(measurements), len(cases))
        by_name = {measurement.case_name: measurement for measurement in measurements}
        self.assertGreater(
            by_name["medium-fast"].system_throughput_req_s,
            by_name["small-fast"].system_throughput_req_s,
        )
        for measurement in measurements:
            self.assertGreaterEqual(measurement.elapsed_s, 0.0)
            self.assertGreater(measurement.system_throughput_req_s, 0.0)
            self.assertGreater(measurement.estimated_latency_s, 0.0)
            self.assertGreaterEqual(measurement.iterations, 1)
            self.assertLessEqual(measurement.iterations, 6)
            for hardware, total in measurement.capacity.items():
                used = sum(
                    worker_alloc.get(hardware, 0)
                    for worker_alloc in measurement.allocation.values()
                )
                self.assertEqual(used, total)

    def test_simulates_autoscaling_changes_over_variable_load_windows(self):
        base_workload = WorkloadProfile.synthetic(
            arrival_rate=1.0,
            short_input=256,
            long_input=1024,
            short_output=64,
            long_output=256,
            samples=12,
            max_batch_size=4,
        )
        windows = [
            LoadWindow("low-5m", start_s=0.0, duration_s=300.0, arrival_rate=0.5),
            LoadWindow(
                "surge-10m",
                start_s=300.0,
                duration_s=600.0,
                arrival_rate=10.0,
                input_scale=1.2,
                output_scale=1.5,
            ),
            LoadWindow("recovery-30m", start_s=900.0, duration_s=1800.0, arrival_rate=1.0),
        ]
        capacity = {"fast": 6}
        framework = make_test_framework(iterations=6)

        measurements = simulate_autoscaling_windows(
            framework,
            base_workload,
            capacity,
            windows,
        )

        self.assertEqual([m.duration_s for m in measurements], [300.0, 600.0, 1800.0])
        self.assertEqual(measurements[0].action_by_worker["ffn"], "initial")
        self.assertIn("scale_up", measurements[1].action_by_worker.values())
        self.assertIn("scale_down", measurements[2].action_by_worker.values())
        self.assertGreater(measurements[1].worker_expansion["ffn"], 1.0)
        self.assertLess(measurements[2].worker_expansion["ffn"], 1.0)
        for index, measurement in enumerate(measurements):
            self.assertGreaterEqual(measurement.elapsed_s, 0.0)
            self.assertGreater(measurement.system_throughput_req_s, 0.0)
            if index > 0:
                self.assertNotEqual(measurement.allocation_delta_by_worker, {})
            used = sum(
                worker_alloc.get("fast", 0)
                for worker_alloc in measurement.allocation.values()
            )
            self.assertGreaterEqual(used, 3)
            self.assertLessEqual(used, capacity["fast"])


if __name__ == "__main__":
    unittest.main()
