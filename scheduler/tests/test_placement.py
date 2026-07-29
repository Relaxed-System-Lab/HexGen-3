import unittest
from unittest.mock import MagicMock, patch
from simulator.core.scheduler import HeterogeneousRiskAwareScheduler, PlacementDecision
from simulator.core.request import GenerationRequest
from simulator.core.engine import ServingEngine

class TestHeterogeneousRiskAwareScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = HeterogeneousRiskAwareScheduler(
            high_intensity_threshold=2048,
            tier3_kv_safe_buffer=0.5
        )

    def create_mock_engine(self, engine_id, hardware, total_memory, model_weights_memory):
        engine = MagicMock(spec=ServingEngine)
        engine.engine_id = engine_id
        engine.hardware = hardware
        engine.model_params = MagicMock()
        engine.model_params.hidden_size = 4096
        engine.model_params.num_hidden_layers = 32
        
        # Mock get_memory_info
        engine.get_memory_info.return_value = {
            'total': total_memory,
            'model_weights': model_weights_memory,
            'kv_cache': 0, # Initially empty
            'available': total_memory - model_weights_memory,
            'used': model_weights_memory
        }
        
        # Mock get_current_load
        engine.get_current_load.return_value = 0
        
        # Mock _estimate_kv_cache_memory
        # Simple mock: 1MB per token for easy math
        # Real calculation is more complex, but we just need to test the logic
        engine._estimate_kv_cache_memory.side_effect = lambda req: req.input_length * 1024 * 1024 
        
        return engine

    def test_tier_detection(self):
        h100 = self.create_mock_engine("e1", "H100-SXM", 80e9, 20e9)
        a100 = self.create_mock_engine("e2", "A100-SXM", 80e9, 20e9)
        rtx3090 = self.create_mock_engine("e3", "RTX 3090", 24e9, 10e9)
        
        self.assertEqual(self.scheduler._get_tier(h100), 1)
        self.assertEqual(self.scheduler._get_tier(a100), 2)
        self.assertEqual(self.scheduler._get_tier(rtx3090), 3)

    def test_hard_constraint_filtering_tier3(self):
        # Tier 3: 24GB total, 10GB weights -> 14GB KV budget.
        # Safe buffer 0.5 -> 7GB max initial footprint.
        # Mock estimate: 1MB per token.
        # Max tokens = 7000.
        
        rtx3090 = self.create_mock_engine("rtx", "RTX 3090", 24 * 1024**3, 10 * 1024**3)
        # KV Budget = 14GB. Safe limit = 7GB.
        
        # Request 1: 5000 tokens -> 5GB -> OK (Threshold is ~5.6GB)
        req1 = GenerationRequest("r1", "model", 5000, 100, 0)
        
        # Request 2: 7000 tokens -> 7GB -> Blocked
        req2 = GenerationRequest("r2", "model", 7000, 100, 0)
        
        # Test Req 1
        decision = self.scheduler.place_request(req1, [rtx3090])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.target_engine, rtx3090)
        
        # Test Req 2
        decision = self.scheduler.place_request(req2, [rtx3090])
        self.assertIsNone(decision)

    def test_arithmetic_intensity_matching(self):
        h100 = self.create_mock_engine("h100", "H100", 80e9, 20e9)
        a100 = self.create_mock_engine("a100", "A100", 80e9, 20e9)
        
        # Threshold is 2048
        
        # High intensity request
        req_high = GenerationRequest("high", "model", 3000, 100, 0)
        decision = self.scheduler.place_request(req_high, [h100, a100])
        self.assertEqual(decision.target_engine, h100)
        
        # Low intensity request -> should go to least loaded/best score
        # Since both are empty, logic falls back to score.
        # Both have same memory, same load.
        # Let's make A100 have slightly better score (more free memory)
        a100.get_memory_info.return_value['model_weights'] = 10e9 # More free space
        
        req_low = GenerationRequest("low", "model", 1000, 100, 0)
        decision = self.scheduler.place_request(req_low, [h100, a100])
        self.assertEqual(decision.target_engine, a100)

    def test_load_balancing(self):
        # Two A100s
        e1 = self.create_mock_engine("e1", "A100", 80e9, 20e9)
        e2 = self.create_mock_engine("e2", "A100", 80e9, 20e9)
        
        # e1 has 10GB KV used
        e1.get_memory_info.return_value['kv_cache'] = 10e9
        
        # e2 has 50GB KV used
        e2.get_memory_info.return_value['kv_cache'] = 50e9
        
        req = GenerationRequest("r1", "model", 1000, 100, 0)
        
        decision = self.scheduler.place_request(req, [e1, e2])
        # Should pick e1 (less pressure)
        self.assertEqual(decision.target_engine, e1)

if __name__ == '__main__':
    unittest.main()
