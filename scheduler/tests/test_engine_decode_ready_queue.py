import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simulator.core.engine import ServingEngine
from simulator.core.request import GenerationRequest


class TestEngineDecodeReadyQueue(unittest.TestCase):
    @patch("simulator.core.engine.ModelAnalyzer")
    @patch("simulator.core.engine.AutoConfig.from_pretrained")
    def test_step_moves_completed_prefill_to_decode_ready_deque(
        self, mock_from_pretrained, mock_model_analyzer
    ):
        mock_from_pretrained.return_value = SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
        )
        mock_model_analyzer.return_value.analyze.return_value = {
            "total_results": {
                "prefill": {"inference_time": 0.001},
                "decode": {"inference_time": 0.001},
            }
        }

        engine = ServingEngine(
            engine_id="n0",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            model_instance=None,
            hardware="NVDA:H100:SXM",
            max_batch_size=8,
            pd_separation=False,
        )

        req = GenerationRequest(
            req_id="r1",
            model="meta-llama/Llama-3.1-8B-Instruct",
            input_length=32,
            output_length=16,
            arrive_at=0.0,
        )

        engine.prefill_queue.append(req)
        engine.can_accommodate_request = lambda _request, safety_margin=0.1: True

        # Keep this test focused on the transition from prefill -> decode-ready.
        engine._form_decode_batch = lambda: None
        engine._add_to_decode_batch = lambda: None

        engine.step(current_time=0.0)

        self.assertIsNone(engine.current_prefill_request)
        self.assertEqual(len(engine.decode_ready_requests), 1)
        self.assertIs(engine.decode_ready_requests[0], req)


if __name__ == "__main__":
    unittest.main()
