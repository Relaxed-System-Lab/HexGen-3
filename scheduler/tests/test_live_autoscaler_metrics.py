import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

from run_hexgen3_live_autoscaler import (  # noqa: E402
    _aggregate_metrics_samples,
    _aggregate_parallel_metrics_samples,
    _plan_existing_allocation,
    _runtime_health_probes,
    _scale_confirmation_signature,
    _scale_in_guard,
    _workload_from_metrics,
)
from simulator.scheduling.types import AllocationMatrix, WorkloadProfile


class TestLiveAutoscalerMetrics(unittest.TestCase):
    def test_runtime_health_probes_wait_for_warmup_readiness(self):
        probes = _runtime_health_probes(
            {
                "runtime_launch": {
                    "processes": [
                        {
                            "name": "prefill-0",
                            "role": "prefill",
                            "command": [
                                "python",
                                "-m",
                                "sglang.launch_server",
                                "--host",
                                "0.0.0.0",
                                "--port",
                                "30001",
                            ],
                        },
                        {
                            "name": "decode-attn-0",
                            "role": "attention",
                            "command": ["python", "--port", "30002"],
                        },
                        {
                            "name": "mini-lb",
                            "role": "lb",
                            "command": ["python", "--port", "30000"],
                        },
                    ]
                }
            }
        )

        self.assertEqual(
            probes,
            {
                "prefill-0": "http://127.0.0.1:30001/health_ready",
                "decode-attn-0": "http://127.0.0.1:30002/health_ready",
            },
        )

    def test_lb_arrival_rate_uses_request_count_over_window_time(self):
        metrics = _aggregate_metrics_samples(
            [
                {
                    "timestamp": 1.0,
                    "worker_type": "lb",
                    "window_s": 2.0,
                    "window_requests": 16,
                    "arrival_rate_rps": 8.0,
                },
                {
                    "timestamp": 2.0,
                    "worker_type": "lb",
                    "window_s": 0.5,
                    "window_requests": 16,
                    "arrival_rate_rps": 32.0,
                },
            ]
        )

        self.assertAlmostEqual(metrics["arrival_rate_rps"], 12.8)

    def test_attention_averages_are_weighted_by_effective_requests(self):
        metrics = _aggregate_metrics_samples(
            [
                {
                    "timestamp": 1.0,
                    "worker_type": "attn",
                    "window_received_requests": 8,
                    "window_finished_requests": 4,
                    "avg_input_tokens": 512.0,
                    "avg_output_tokens": 128.0,
                },
                {
                    "timestamp": 2.0,
                    "worker_type": "attn",
                    "window_received_requests": 24,
                    "window_finished_requests": 12,
                    "avg_input_tokens": 1024.0,
                    "avg_output_tokens": 512.0,
                },
            ]
        )

        self.assertAlmostEqual(metrics["avg_input_tokens"], 896.0)
        self.assertAlmostEqual(metrics["avg_output_tokens"], 416.0)

    def test_output_shape_keeps_running_progress_separate_from_completions(self):
        metrics = _aggregate_metrics_samples(
            [
                {
                    "timestamp": 1.0,
                    "worker_type": "attn",
                    "window_finished_requests": 2,
                    "window_failed_requests": 5,
                    "window_finished_output_tokens": 256,
                    "avg_output_tokens": 128.0,
                    "running_requests": 3,
                    "running_output_tokens": 9999,
                },
                {
                    "timestamp": 2.0,
                    "worker_type": "attn",
                    "window_finished_requests": 2,
                    "window_failed_requests": 7,
                    "window_finished_output_tokens": 1024,
                    "avg_output_tokens": 512.0,
                    "running_requests": 4,
                    "running_output_tokens": 1200,
                },
            ]
        )

        self.assertEqual(metrics["aggregated_finished_requests"], 4)
        self.assertEqual(metrics["aggregated_failed_requests"], 12)
        self.assertEqual(metrics["aggregated_finished_output_tokens"], 1280)
        self.assertEqual(metrics["running_requests"], 4)
        self.assertEqual(metrics["running_output_tokens"], 1200)
        self.assertAlmostEqual(metrics["completed_avg_output_tokens"], 320.0)
        self.assertAlmostEqual(metrics["avg_running_output_tokens"], 300.0)
        self.assertAlmostEqual(metrics["avg_output_tokens"], 320.0)

    def test_output_shape_uses_running_progress_when_nothing_has_finished(self):
        metrics = _aggregate_metrics_samples(
            [
                {
                    "timestamp": 1.0,
                    "worker_type": "attn",
                    "window_finished_requests": 0,
                    "window_finished_output_tokens": 0,
                    "running_requests": 4,
                    "running_output_tokens": 1200,
                }
            ]
        )

        self.assertEqual(metrics["completed_avg_output_tokens"], 0.0)
        self.assertEqual(metrics["avg_running_output_tokens"], 300.0)
        self.assertEqual(metrics["avg_output_tokens"], 300.0)

    def test_parallel_attention_metrics_keep_direct_request_averages(self):
        metrics = _aggregate_parallel_metrics_samples(
            [
                {
                    "timestamp": 3.0,
                    "worker_type": "lb",
                    "arrival_rate_rps": 32.0,
                },
                {
                    "timestamp": 3.0,
                    "worker_type": "attn",
                    "avg_input_tokens": 512.0,
                    "avg_output_tokens": 128.0,
                    "aggregated_received_requests": 8,
                    "aggregated_finished_requests": 4,
                },
                {
                    "timestamp": 3.0,
                    "worker_type": "attn",
                    "avg_input_tokens": 1024.0,
                    "avg_output_tokens": 512.0,
                    "aggregated_received_requests": 24,
                    "aggregated_finished_requests": 12,
                },
            ],
            raw_sample_count=3,
        )

        workload = _workload_from_metrics(None, metrics)
        self.assertEqual(workload.arrival_rate, 32.0)
        self.assertEqual(workload.mean_input, 896)
        self.assertEqual(workload.mean_output, 416)

    def test_workload_profile_uses_direct_attention_averages(self):
        workload = _workload_from_metrics(
            None,
            {
                "arrival_rate_rps": 32.0,
                "avg_input_tokens": 512.0,
                "avg_output_tokens": 1024.0,
            },
        )

        self.assertEqual(workload.mean_input, 512)
        self.assertEqual(workload.mean_output, 1024)

    def test_workload_profile_reuses_recent_independent_shape_observations(self):
        args = SimpleNamespace(workload_shape_max_age_s=60.0)
        shape_state = {}
        _workload_from_metrics(
            args,
            {
                "arrival_rate_rps": 8.0,
                "avg_input_tokens": 512.0,
                "avg_output_tokens": 128.0,
            },
            shape_state=shape_state,
            now_s=100.0,
        )

        workload = _workload_from_metrics(
            args,
            {
                "arrival_rate_rps": 32.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 1024.0,
            },
            shape_state=shape_state,
            now_s=120.0,
        )
        self.assertEqual(workload.mean_input, 512)
        self.assertEqual(workload.mean_output, 1024)

        workload = _workload_from_metrics(
            args,
            {
                "arrival_rate_rps": 16.0,
                "avg_input_tokens": 256.0,
                "avg_output_tokens": 0.0,
            },
            shape_state=shape_state,
            now_s=130.0,
        )
        self.assertEqual(workload.mean_input, 256)
        self.assertEqual(workload.mean_output, 1024)

    def test_running_output_is_a_non_decreasing_lower_bound_until_completion(self):
        args = SimpleNamespace(workload_shape_max_age_s=300.0)
        shape_state = {}

        def workload_for(metrics, now_s):
            return _workload_from_metrics(
                args,
                {
                    "arrival_rate_rps": 32.0,
                    "avg_input_tokens": 1024.0,
                    **metrics,
                },
                shape_state=shape_state,
                now_s=now_s,
            )

        first = workload_for(
            {
                "avg_output_tokens": 213.0,
                "aggregated_finished_requests": 0,
                "running_requests": 100,
                "avg_running_output_tokens": 213.0,
            },
            100.0,
        )
        rising = workload_for(
            {
                "avg_output_tokens": 402.0,
                "aggregated_finished_requests": 0,
                "running_requests": 100,
                "avg_running_output_tokens": 402.0,
            },
            130.0,
        )
        replacement_batch = workload_for(
            {
                "avg_output_tokens": 299.0,
                "aggregated_finished_requests": 0,
                "running_requests": 100,
                "avg_running_output_tokens": 299.0,
            },
            160.0,
        )
        completed = workload_for(
            {
                "avg_output_tokens": 512.0,
                "completed_avg_output_tokens": 512.0,
                "aggregated_finished_requests": 20,
                "running_requests": 100,
                "avg_running_output_tokens": 310.0,
            },
            190.0,
        )
        shorter_completed = workload_for(
            {
                "avg_output_tokens": 128.0,
                "completed_avg_output_tokens": 128.0,
                "aggregated_finished_requests": 20,
                "running_requests": 100,
                "avg_running_output_tokens": 70.0,
            },
            220.0,
        )

        self.assertEqual(first.mean_output, 213)
        self.assertEqual(rising.mean_output, 402)
        self.assertEqual(replacement_batch.mean_output, 402)
        self.assertEqual(completed.mean_output, 512)
        self.assertEqual(shorter_completed.mean_output, 128)

    def test_workload_shape_cache_expires(self):
        args = SimpleNamespace(workload_shape_max_age_s=10.0)
        shape_state = {}
        _workload_from_metrics(
            args,
            {
                "arrival_rate_rps": 8.0,
                "avg_input_tokens": 512.0,
                "avg_output_tokens": 128.0,
            },
            shape_state=shape_state,
            now_s=100.0,
        )

        with self.assertRaisesRegex(ValueError, "input tokens"):
            _workload_from_metrics(
                args,
                {
                    "arrival_rate_rps": 8.0,
                    "avg_input_tokens": 0.0,
                    "avg_output_tokens": 0.0,
                },
                shape_state=shape_state,
                now_s=111.0,
            )

    def test_cached_output_without_new_evidence_does_not_refresh_its_age(self):
        args = SimpleNamespace(workload_shape_max_age_s=10.0)
        shape_state = {}
        _workload_from_metrics(
            args,
            {
                "arrival_rate_rps": 8.0,
                "avg_input_tokens": 512.0,
                "avg_output_tokens": 128.0,
            },
            shape_state=shape_state,
            now_s=100.0,
        )
        cached = _workload_from_metrics(
            args,
            {
                "arrival_rate_rps": 8.0,
                "avg_input_tokens": 512.0,
                "avg_output_tokens": 0.0,
            },
            shape_state=shape_state,
            now_s=105.0,
        )
        self.assertEqual(cached.mean_output, 128)

        with self.assertRaisesRegex(ValueError, "output tokens"):
            _workload_from_metrics(
                args,
                {
                    "arrival_rate_rps": 8.0,
                    "avg_input_tokens": 512.0,
                    "avg_output_tokens": 0.0,
                },
                shape_state=shape_state,
                now_s=111.0,
            )

    def test_scale_decision_re_evaluates_current_allocation_for_new_workload(self):
        allocation = AllocationMatrix.zeros(("gpu",))
        for worker in ("pre", "attn", "ffn"):
            allocation.set(worker, "gpu", 1)
        previous_plan = SimpleNamespace(
            allocation=allocation,
            parallelism={"old": "parallelism"},
        )
        refreshed_plan = SimpleNamespace(
            allocation=allocation.clone(),
            parallelism={"new": "parallelism"},
        )
        workload = WorkloadProfile(
            arrival_rate=8.0,
            input_lengths=(512,),
            output_lengths=(128,),
        )

        class FakeFramework:
            def __init__(self):
                self.scaled_from = None

            def evaluate_allocation(
                self,
                current_workload,
                current_allocation,
                previous_parallelism=None,
            ):
                self.asserted = (
                    current_workload,
                    current_allocation,
                    previous_parallelism,
                )
                return refreshed_plan

            def proportional_scale_allocation(self, current_workload, plan, capacity):
                self.scaled_from = plan
                return plan.allocation.clone()

            def worker_expansion_factors(self, current_workload, plan):
                return {"pre": 1.0, "attn": 1.0, "ffn": 1.0}

            def reschedule(self, current_workload, plan, capacity):
                raise AssertionError("unchanged refreshed allocation must not reschedule")

        framework = FakeFramework()
        plan, decision, _ = _plan_existing_allocation(
            framework,
            workload,
            previous_plan,
            {"gpu": 8},
        )

        self.assertIs(plan, refreshed_plan)
        self.assertIs(framework.scaled_from, refreshed_plan)
        self.assertEqual(decision, "hold_allocation_unchanged")

    def test_final_scale_decision_uses_rescheduled_allocation(self):
        current = SimpleNamespace(
            allocation=self._allocation(pre=2, attn=2, ffn=2),
            parallelism={},
        )
        raw_target = self._allocation(pre=2, attn=1, ffn=2)

        class FakeFramework:
            def evaluate_allocation(self, *args, **kwargs):
                return current

            def proportional_scale_allocation(self, *args, **kwargs):
                return raw_target

            def worker_expansion_factors(self, *args, **kwargs):
                return {"pre": 1.0, "attn": 0.5, "ffn": 1.0}

            def reschedule(self, *args, **kwargs):
                return current

        framework = FakeFramework()
        workload = WorkloadProfile(
            arrival_rate=28.0,
            input_lengths=(1024,),
            output_lengths=(384,),
        )

        plan, decision, _ = _plan_existing_allocation(
            framework,
            workload,
            current,
            {"gpu": 8},
            allow_scale_in=False,
        )

        self.assertIs(plan, current)
        self.assertEqual(decision, "hold_allocation_unchanged")

    def test_large_waiting_backlog_blocks_scale_in(self):
        args = SimpleNamespace(
            poll_interval_s=30.0,
            scale_in_backlog_threshold=None,
        )
        workload = WorkloadProfile(
            arrival_rate=32.0,
            input_lengths=(1024,),
            output_lengths=(384,),
        )

        allowed, waiting, threshold = _scale_in_guard(
            args,
            workload,
            {"waiting_requests": 1200},
        )

        self.assertFalse(allowed)
        self.assertEqual(waiting, 1200)
        self.assertEqual(threshold, 960)

    def test_scale_out_confirmation_survives_changing_target_allocations(self):
        current = SimpleNamespace(
            allocation=self._allocation(pre=1, attn=1, ffn=1),
            parallelism={},
        )
        smaller_target = SimpleNamespace(
            allocation=self._allocation(pre=2, attn=1, ffn=2),
            parallelism={},
        )
        larger_target = SimpleNamespace(
            allocation=self._allocation(pre=2, attn=2, ffn=4),
            parallelism={},
        )

        self.assertEqual(
            _scale_confirmation_signature(current, smaller_target),
            _scale_confirmation_signature(current, larger_target),
        )

    def test_scale_in_confirmation_still_requires_the_same_target(self):
        current = SimpleNamespace(
            allocation=self._allocation(pre=2, attn=2, ffn=4),
            parallelism={},
        )
        first_target = SimpleNamespace(
            allocation=self._allocation(pre=1, attn=1, ffn=2),
            parallelism={},
        )
        second_target = SimpleNamespace(
            allocation=self._allocation(pre=1, attn=1, ffn=1),
            parallelism={},
        )

        self.assertNotEqual(
            _scale_confirmation_signature(current, first_target),
            _scale_confirmation_signature(current, second_target),
        )

    @staticmethod
    def _allocation(*, pre, attn, ffn):
        allocation = AllocationMatrix.zeros(("gpu",))
        allocation.set("pre", "gpu", pre)
        allocation.set("attn", "gpu", attn)
        allocation.set("ffn", "gpu", ffn)
        return allocation


if __name__ == "__main__":
    unittest.main()
