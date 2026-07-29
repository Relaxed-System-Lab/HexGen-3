import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from cluster_experiments.common.load_balancer import LoadBalancer, Backend

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_collect_metrics(self):
        async def run_test():
            # Mock backends
            backend1 = MagicMock(spec=Backend)
            backend1.name = "backend1"
            backend1.url = "http://backend1"
            
            backend2 = MagicMock(spec=Backend)
            backend2.name = "backend2"
            backend2.url = "http://backend2"
            
            # Mock responses
            response1 = MagicMock()
            response1.status_code = 200
            response1.text = 'http_requests_total{method="GET",engine="0"} 10\n# HELP something\nsomething_else 5'
            
            response2 = MagicMock()
            response2.status_code = 200
            response2.text = 'http_requests_total{method="POST",engine="0"} 20'
            
            # Mock client
            client = AsyncMock()
            client.get.side_effect = [response1, response2]
            
            # Initialize LoadBalancer with mocks
            with patch('cluster_experiments.common.load_balancer.httpx.AsyncClient', return_value=client):
                lb = LoadBalancer(config={'scheduler_type': 'round_robin'})
                lb.backends = [backend1, backend2]
                lb.client = client
                
                # Test text metrics
                metrics = await lb.collect_metrics()
                print("Collected Metrics:")
                print(metrics)
                
                self.assertIn('http_requests_total{backend="backend1",method="GET",engine="backend1"} 10', metrics)
                self.assertIn('something_else{backend="backend1"} 5', metrics)
                self.assertIn('http_requests_total{backend="backend2",method="POST",engine="backend2"} 20', metrics)
                
                # Test JSON metrics
                json_metrics = await lb.collect_metrics_json()
                print("\nCollected JSON Metrics:")
                print(json_metrics)
                
                # Verify JSON structure
                self.assertTrue(any(
                    m['name'] == 'http_requests_total' and 
                    m['labels']['backend'] == 'backend1' and 
                    m['labels']['engine'] == 'backend1' and 
                    m['value'] == 10.0 
                    for m in json_metrics
                ))
                
                self.assertTrue(any(
                    m['name'] == 'something_else' and 
                    m['labels']['backend'] == 'backend1' and 
                    m['value'] == 5.0 
                    for m in json_metrics
                ))

        self.loop.run_until_complete(run_test())

if __name__ == "__main__":
    unittest.main()
