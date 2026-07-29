import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

from _autoscaling_stability import StabilityValidationConfig  # noqa: E402
from run_hexgen3_live_autoscaler import _plan_existing_allocation  # noqa: E402
from simulator.scheduling import (  # noqa: E402
    AllocationMatrix,
    AutoscalingConfig,
    HexGenSchedulingFramework,
    LocalSchedulerConfig,
    WorkloadProfile,
)


class OscillatingEstimator:
    throughput_by_worker_gpus = {
        "pre": {1: 20.0, 2: 60.0},
        "attn": {1: 20.0, 2: 80.0, 4: 160.0},
        "ffn": {1: 10.0, 2: 45.0, 4: 160.0},
    }

    def estimate_slice_throughput(self, worker_type, hardware, strategy, workload):
        values = self.throughput_by_worker_gpus[worker_type]
        if strategy.gpus in values:
            return values[strategy.gpus]
        largest = max(values)
        return values[largest] * strategy.gpus / largest

    def estimate_latency(self, workload, throughput):
        return 1.0 / max(throughput.bottleneck, 1e-9)

    def estimate_tail_latency(self, workload, throughput):
        latency = self.estimate_latency(workload, throughput)
        return {"mean": latency, "p50": latency, "p95": latency, "p99": latency}

    def estimate_cost_per_hour(self, allocation):
        return float(allocation.total_gpus())


class TestLiveAutoscalerStability(unittest.TestCase):
    validation_config = StabilityValidationConfig(
        max_rounds=3,
        max_candidates=16,
        timeout_s=5.0,
    )

    def test_stability_validation_breaks_proportional_scaling_cycle(self):
        framework = self._framework()
        workload = self._workload()
        capacity = {"fast": 8}
        current = framework.evaluate_allocation(
            workload,
            self._allocation(pre=1, attn=1, ffn=1),
        )

        plan, decision, _ = _plan_existing_allocation(
            framework,
            workload,
            current,
            capacity,
            validation_config=self.validation_config,
        )

        self.assertEqual(decision, "scale_changed")
        self.assertEqual(self._worker_totals(plan.allocation), (2, 2, 2))
        validation = plan.metadata["autoscaling"]["stability_validation"]
        self.assertTrue(validation["triggered"])
        self.assertEqual(validation["selection"], "quantization_hold")
        self.assertGreaterEqual(
            plan.throughput.bottleneck,
            validation["required_safe_throughput_req_s"],
        )

        next_plan, next_decision, _ = _plan_existing_allocation(
            framework,
            workload,
            plan,
            capacity,
            validation_config=self.validation_config,
        )
        self.assertEqual(next_decision, "hold_allocation_unchanged")
        self.assertEqual(next_plan.allocation.as_key(), plan.allocation.as_key())

    def test_capacity_limited_fallback_holds_highest_throughput_plan(self):
        framework = self._framework()
        workload = self._workload()
        capacity = {"fast": 4}
        current = framework.evaluate_allocation(
            workload,
            self._allocation(pre=1, attn=1, ffn=1),
        )

        plan, _, _ = _plan_existing_allocation(
            framework,
            workload,
            current,
            capacity,
            validation_config=self.validation_config,
        )
        validation = plan.metadata["autoscaling"]["stability_validation"]

        self.assertEqual(validation["selection"], "capacity_limited")
        self.assertEqual(self._worker_totals(plan.allocation), (1, 1, 2))
        self.assertLess(
            plan.throughput.bottleneck,
            validation["required_safe_throughput_req_s"],
        )

        next_plan, next_decision, _ = _plan_existing_allocation(
            framework,
            workload,
            plan,
            capacity,
            validation_config=self.validation_config,
        )
        self.assertEqual(next_decision, "hold_allocation_unchanged")
        self.assertEqual(next_plan.allocation.as_key(), plan.allocation.as_key())

    def test_backlog_guard_prevents_any_worker_scale_in(self):
        framework = self._framework()
        workload = self._workload()
        capacity = {"fast": 8}
        current = framework.evaluate_allocation(
            workload,
            self._allocation(pre=2, attn=2, ffn=4),
        )

        plan, decision, _ = _plan_existing_allocation(
            framework,
            workload,
            current,
            capacity,
            allow_scale_in=False,
            validation_config=self.validation_config,
        )

        self.assertEqual(decision, "hold_allocation_unchanged")
        self.assertEqual(plan.allocation.as_key(), current.allocation.as_key())
        self.assertEqual(
            plan.metadata["autoscaling"]["stability_validation"]["selection"],
            "backlog_hold",
        )

    @staticmethod
    def _framework():
        framework = HexGenSchedulingFramework(
            local_config=LocalSchedulerConfig(
                node_gpus=8,
                match_attn_ffn_dp=False,
            ),
            autoscaling_config=AutoscalingConfig(
                target_utilization=0.75,
                hysteresis=0.08,
                min_scale_factor=0.0,
                max_scale_factor=float("inf"),
                decode_worker_gpu_choices=(1, 2, 4, 8),
            ),
        )
        framework.estimator = OscillatingEstimator()
        framework.local_scheduler.estimator = framework.estimator
        return framework

    @staticmethod
    def _workload():
        return WorkloadProfile(
            arrival_rate=28.0,
            input_lengths=(1024,),
            output_lengths=(384,),
        )

    @staticmethod
    def _allocation(*, pre, attn, ffn):
        allocation = AllocationMatrix.zeros(("fast",))
        allocation.set("pre", "fast", pre)
        allocation.set("attn", "fast", attn)
        allocation.set("ffn", "fast", ffn)
        return allocation

    @staticmethod
    def _worker_totals(allocation):
        return tuple(
            sum(allocation.values[worker].values())
            for worker in ("pre", "attn", "ffn")
        )


if __name__ == "__main__":
    unittest.main()
