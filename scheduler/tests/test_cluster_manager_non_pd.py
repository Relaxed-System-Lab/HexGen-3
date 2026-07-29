import unittest
from unittest.mock import patch

from simulator.core.arrival import PoissonProcess
from simulator.core.cluster_manager import (
    ClusterConfiguration,
    ClusterManager,
    NodeConfiguration,
)
from simulator.core.events import Event, EventPriority, EventType
from simulator.core.request import GenerationRequest
from simulator.core.scheduler import PlacementDecision


class _FakeEngine:
    def __init__(self, **kwargs):
        self.engine_id = kwargs["engine_id"]
        self.pd_prefill_only = kwargs.get("pd_prefill_only", False)
        self.pd_decode_only = kwargs.get("pd_decode_only", False)
        self.pd_separation = kwargs.get("pd_separation", False)
        self.prefill_queue = []
        self.current_prefill_request = None
        self.decode_ready_requests = []
        self.current_decode_batch = None
        self.time_cursor = 0.0
        self.hardware_spec = {"vmemory": 1.0}

    def set_event_callback(self, callback):
        self._callback = callback

    def add_request(self, request):
        return True

    def add_decode_ready_request(self, request, ready_time=0.0):
        return True

    def get_current_load(self):
        return 0

    def get_memory_info(self):
        return {"used": 0.0, "utilization": 0.0}

    def set_current_time(self, current_time):
        self.time_cursor = current_time


class TestClusterManagerNonPDPlacement(unittest.TestCase):
    def test_non_pd_prefill_uses_all_online_engines(self):
        cfg = ClusterConfiguration(
            cluster_id="c1",
            nodes=[
                NodeConfiguration(
                    node_id="n1",
                    model_id="m",
                    hardware="NVDA:RTX3090",
                    pd_separation=False,
                    pd_prefill_only=False,
                    pd_decode_only=False,
                ),
                NodeConfiguration(
                    node_id="n2",
                    model_id="m",
                    hardware="NVDA:RTX3090",
                    pd_separation=False,
                    pd_prefill_only=False,
                    pd_decode_only=False,
                ),
            ],
            scheduler_algorithm="random",
        )

        with patch("simulator.core.cluster_manager.ServingEngine", _FakeEngine):
            mgr = ClusterManager(cfg, PoissonProcess(arrival_rate=1.0))

        req = GenerationRequest(
            req_id="r1",
            model="m",
            input_length=16,
            output_length=8,
            arrive_at=0.0,
        )
        mgr.active_requests[req.req_id] = req

        seen = {"count": 0}

        def fake_place_request(request, available_engines):
            seen["count"] = len(available_engines)
            return PlacementDecision(
                target_engine=available_engines[0],
                reason="test",
            )

        mgr.scheduler.place_request = fake_place_request

        ev = Event(
            timestamp=0.0,
            event_type=EventType.PLACEMENT_DECISION,
            target="cluster_manager",
            data={"request_id": "r1", "phase": "prefill"},
            priority=EventPriority.MEDIUM,
        )
        mgr._handle_placement_decision(ev)

        self.assertEqual(seen["count"], 2)
        self.assertTrue(req.prefill_node in {"n1", "n2"})
        self.assertFalse(
            any(r.get("reason") == "no_prefill_engine_available" for r in mgr.rejected_requests)
        )


if __name__ == "__main__":
    unittest.main()
