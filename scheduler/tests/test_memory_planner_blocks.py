import unittest
from types import SimpleNamespace

from simulator.configs.hardware import hardware_params
from simulator.configs.models import llama
from simulator.core.memory import MemoryPlanner
from simulator.core.request import GenerationRequest


class TestMemoryPlannerBlocks(unittest.TestCase):
    def test_additional_blocks_account_for_prompt_plus_generated_tokens(self):
        model_params = SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
        )
        planner = MemoryPlanner(
            model_params=model_params,
            model_config=llama,
            hardware_params=hardware_params["NVDA:H100:SXM"],
            block_size=16,
        )

        req = GenerationRequest(
            req_id="r1",
            model="meta-llama/Llama-3.1-8B-Instruct",
            input_length=32,
            output_length=16,
            arrive_at=0.0,
        )

        planner.allocate(req)

        # Prompt length=32 already reserves 2 blocks.
        # Once one token is generated, total sequence is 33 and a new block is required.
        req.generated_tokens = 1
        additional = planner._estimate_required_blocks(req)

        self.assertEqual(additional, 1)


if __name__ == "__main__":
    unittest.main()
