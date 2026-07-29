import unittest
from types import SimpleNamespace

from sglang.srt.metrics.afd_metrics_collector import AFDMetricsCollector, AFDMetricsSample


class TestAFDMetricsCollector(unittest.TestCase):
    def test_serialization_preserves_float_metrics_and_window_counts(self):
        sample = AFDMetricsSample(
            timestamp=123.75,
            worker_type="attn",
            workload_uncached_tokens=10.5,
            window_s=1.25,
            window_received_requests=8,
            window_finished_requests=4,
            window_failed_requests=2,
            window_finished_output_tokens=513,
            arrival_rate_rps=6.4,
            avg_input_tokens=512.5,
            avg_output_tokens=128.25,
            running_requests=3,
            running_output_tokens=96,
            avg_running_output_tokens=32.0,
        )

        serialized = sample.to_dict()

        self.assertEqual(serialized["timestamp"], 123.75)
        self.assertEqual(serialized["window_s"], 1.25)
        self.assertEqual(serialized["arrival_rate_rps"], 6.4)
        self.assertEqual(serialized["avg_input_tokens"], 512.5)
        self.assertEqual(serialized["avg_output_tokens"], 128.25)
        self.assertEqual(serialized["window_received_requests"], 8)
        self.assertEqual(serialized["window_finished_requests"], 4)
        self.assertEqual(serialized["window_failed_requests"], 2)
        self.assertEqual(serialized["window_finished_output_tokens"], 513)
        self.assertEqual(serialized["running_requests"], 3)
        self.assertEqual(serialized["running_output_tokens"], 96)
        self.assertEqual(serialized["avg_running_output_tokens"], 32.0)

    def test_running_output_shape_uses_current_unfinished_requests_only(self):
        active = SimpleNamespace(
            rid="active",
            output_ids=[1, 2, 3],
            finished=lambda: False,
        )
        finished = SimpleNamespace(
            rid="finished",
            output_ids=list(range(10)),
            finished=lambda: True,
        )
        health = SimpleNamespace(
            rid="HEALTH_CHECK_1",
            output_ids=list(range(20)),
            finished=lambda: False,
        )
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[active, finished, health]),
        )
        collector = AFDMetricsCollector(
            scheduler,
            "/tmp/test_afd_running_output_shape.jsonl",
            worker_type="attn",
        )

        self.assertEqual(collector._get_running_output_shape(), (1, 3, 3.0))

    def test_waiting_requests_include_disaggregated_queues(self):
        scheduler = SimpleNamespace(
            waiting_queue=[object()],
            grammar_queue=[object()],
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[object(), object()],
                retracted_queue=[object()],
            ),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[object()]),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                queue=[object(), object()],
            ),
            disagg_prefill_inflight_queue=[object(), object(), object()],
        )
        collector = AFDMetricsCollector(
            scheduler,
            "/tmp/test_afd_waiting_requests.jsonl",
            worker_type="attn",
        )

        self.assertEqual(collector._get_waiting_requests(), 11)


if __name__ == "__main__":
    unittest.main()
