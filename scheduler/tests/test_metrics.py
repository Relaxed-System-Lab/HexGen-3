import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cluster_experiments.common.load_balancer import LoadBalancer, Backend

@pytest.mark.asyncio
async def test_collect_metrics():
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
    response1.text = 'http_requests_total{method="GET"} 10\n# HELP something\nsomething_else 5'
    
    response2 = MagicMock()
    response2.status_code = 200
    response2.text = 'http_requests_total{method="POST"} 20'
    
    # Mock client
    client = AsyncMock()
    client.get.side_effect = [response1, response2]
    
    # Initialize LoadBalancer with mocks
    with patch('cluster_experiments.common.load_balancer.httpx.AsyncClient', return_value=client):
        lb = LoadBalancer(config={'scheduler_type': 'round_robin'})
        lb.backends = [backend1, backend2]
        lb.client = client
        
        metrics = await lb.collect_metrics()
        
        print("Collected Metrics:")
        print(metrics)
        
        assert 'http_requests_total{backend="backend1",method="GET"} 10' in metrics
        assert 'something_else{backend="backend1"} 5' in metrics
        assert 'http_requests_total{backend="backend2",method="POST"} 20' in metrics

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_collect_metrics())
