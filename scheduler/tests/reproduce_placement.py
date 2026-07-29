
import logging
from simulator.core.placement import PlacementDecisionMaker, PhysicalNodeConfig, WorkloadConfig
from simulator.core.arrival import PoissonProcess
from simulator.configs.hardware import hardware_params

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_placement_strategies():
    # Define Hardware
    physical_nodes = [
        PhysicalNodeConfig(gpu_type="NVDA:A100", count=2, gpus_per_node=1),
        PhysicalNodeConfig(gpu_type="NVDA:RTX3090", count=4, gpus_per_node=1)
    ]
    
    # Define Workloads
    # Workload A: High load (10 req/s, 1000 tokens) -> Score = 10000
    # Workload B: Low load (1 req/s, 1000 tokens) -> Score = 1000
    # Ratio 10:1
    
    wl_a = WorkloadConfig(
        model_id="meta-llama/Llama-2-7b-hf",
        arrival_process=PoissonProcess(10.0),
        duration=100,
        input_dist=("Constant", [500]),
        output_dist=("Constant", [500])
    )
    
    wl_b = WorkloadConfig(
        model_id="meta-llama/Llama-2-13b-hf",
        arrival_process=PoissonProcess(1.0),
        duration=100,
        input_dist=("Constant", [500]),
        output_dist=("Constant", [500])
    )
    
    workloads = [wl_a, wl_b]
    
    print("\n--- Testing WorkloadBalancedPlacementStrategy ---")
    maker = PlacementDecisionMaker("workload_balanced")
    nodes, metadata = maker.place(physical_nodes, workloads)
    
    # Count assignments
    counts = {wl.model_id: 0 for wl in workloads}
    for node in nodes:
        counts[node.model_id] += 1
        
    print(f"Assignments: {counts}")
    # Expect A to have much more than B
    if counts[wl_a.model_id] > counts[wl_b.model_id]:
        print("PASS: Workload A has more nodes than B.")
    else:
        print("FAIL: Workload A should have more nodes.")

    print("\n--- Testing SimulationSearchPlacementStrategy ---")
    # Use a shorter duration for speed
    maker_sim = PlacementDecisionMaker("simulation_search")
    # We need to monkeypatch the strategy's simulation duration to be very short for this test
    maker_sim.strategy.simulation_duration = 1.0 
    
    nodes_sim, metadata_sim = maker_sim.place(physical_nodes, workloads)
    
    counts_sim = {wl.model_id: 0 for wl in workloads}
    for node in nodes_sim:
        counts_sim[node.model_id] += 1
        
    print(f"Assignments (Sim Search): {counts_sim}")
    if len(nodes_sim) > 0:
        print("PASS: Simulation search returned a configuration.")
    else:
        print("FAIL: Simulation search returned empty.")

if __name__ == "__main__":
    test_placement_strategies()
