import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simulator.core.engine import ServingEngine


class TestEngineMemoryPlannerMode(unittest.TestCase):
    @patch("simulator.core.engine.ModelAnalyzer")
    @patch("simulator.core.engine.AutoConfig.from_pretrained")
    def test_engine_enables_memory_planner_for_llama_fallback_config(
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
        )

        self.assertIsNotNone(engine.memory_planner)


if __name__ == "__main__":
    unittest.main()
