#!/usr/bin/env python3
"""
Example script demonstrating the event-driven cluster simulation.

This script shows how to set up and run a multi-node LLM serving cluster
simulation with different scheduling algorithms and workload patterns.
"""

import os
import sys
import re
import json
import yaml
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.core.cluster_manager import ClusterManager, ClusterConfiguration, NodeConfiguration
from simulator.core.arrival import PoissonProcess, GammaProcess, ArrivalProcess
from simulator.core.events import Event, EventType, EventPriority
from simulator.core.trace import export_chrome_trace_from_results
from simulator.configs.hardware import hardware_params
from simulator.core.config import WorkloadConfig
from simulator.core.placement import PlacementDecisionMaker, PhysicalNodeConfig

def parse_distribution(dist_str: str) -> Tuple[str, List[float]]:
    """Parse distribution string like 'Poisson(5)' or 'Normal(512, 50)'."""
    match = re.match(r"(\w+)\((.*)\)", dist_str)
    if not match:
        raise ValueError(f"Invalid distribution format: {dist_str}")
    
    dist_type = match.group(1)
    params = [float(p.strip()) for p in match.group(2).split(",")]
    return dist_type, params

def create_arrival_process(dist_type: str, params: List[float]) -> ArrivalProcess:
    """Create an arrival process from parsed parameters."""
    if dist_type == "Poisson":
        return PoissonProcess(arrival_rate=params[0])
    elif dist_type == "Gamma":
        return GammaProcess(arrival_rate=params[0], cv=params[1])
    else:
        raise ValueError(f"Unknown arrival process type: {dist_type}")

def sample_from_distribution(dist_type: str, params: List[float]) -> int:
    """Sample a value from the specified distribution."""
    if dist_type == "Normal":
        val = np.random.normal(params[0], params[1])
        return max(1, int(val))
    elif dist_type == "Uniform":
        val = np.random.uniform(params[0], params[1])
        return max(1, int(val))
    elif dist_type == "Constant":
        return int(params[0])
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def load_config(config_path: str) -> Tuple[ClusterConfiguration, List[WorkloadConfig], Dict[str, Any]]:
    """Load cluster and workload configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Parse workload config first to get valid model IDs
    workloads = []
    for wl_config in config_data.get('workload', []):
        arrival_dist, arrival_params = parse_distribution(wl_config['arrival_rate'])
        arrival_process = create_arrival_process(arrival_dist, arrival_params)
        
        input_dist = parse_distribution(wl_config['input'])
        output_dist = parse_distribution(wl_config['output'])
        
        workload = WorkloadConfig(
            model_id=wl_config['model'],
            arrival_process=arrival_process,
            duration=float(wl_config['duration']),
            input_dist=input_dist,
            output_dist=output_dist
        )
        workloads.append(workload)

    default_model_id = workloads[0].model_id if workloads else "meta-llama/Llama-2-7b-hf"

    # Parse hardware/cluster config
    physical_nodes = []
    for node_config in config_data.get('nodes', []):
        physical_nodes.append(PhysicalNodeConfig(
            gpu_type=node_config['gpu'],
            count=node_config['count'],
            gpus_per_node=node_config.get('gpus_per_node', 1), # Default to 1 if not specified
            cost=node_config.get('cost', 0.0)
        ))
        
    placement_strategy = config_data.get('placement_strategy', 'maximize_replicas')
    
    # Parse placement config
    placement_config = config_data.get('placement_config', {})
    # Also support top-level memory_threshold for backward compatibility or convenience
    memory_threshold = placement_config.get('memory_threshold', config_data.get('memory_threshold', 0.8))
    
    placement_maker = PlacementDecisionMaker(placement_strategy, memory_threshold=float(memory_threshold))
    
    # Place replicas for all workloads
    nodes, placement_metadata = placement_maker.place(physical_nodes, workloads)

    cluster_config = ClusterConfiguration(
        cluster_id="yaml_configured_cluster",
        nodes=nodes,
        scheduler_algorithm="random" # Default, can be made configurable
    )

    return cluster_config, workloads, placement_metadata

class MultiModelClusterManager(ClusterManager):
    """Extended ClusterManager to support multiple models and workloads."""
    
    def __init__(self, config: ClusterConfiguration, workloads: List[WorkloadConfig]):
        self.workloads = workloads
        # We pass the first workload's arrival process just to satisfy the super init,
        # but we will override schedule_request_arrivals to use all workloads.
        super().__init__(config, workloads[0].arrival_process)

    def _initialize_cluster(self):
        """Initialize cluster nodes. Overridden to handle model assignment."""
        # In this simplified multi-model simulation, we might want nodes to be able to serve
        # any model, or specific models. The base implementation assigns one model per node.
        # For now, we'll keep the base implementation but note that in a real system
        # we'd need more complex model management (loading/unloading).
        # We will assume for this simulation that nodes are "ready" for the models 
        # or the scheduler handles it.
        
        # To make it work with the existing ServingEngine which takes a model_id,
        # we might need to assume nodes are homogeneous regarding the model they start with,
        # or we update them later.
        # Let's just call super() and let it set up.
        super()._initialize_cluster()
        
        # If we want to simulate a cluster where nodes can serve different models,
        # we might need to update the engines here. But for now, let's assume
        # the scheduler will handle routing or rejection if model doesn't match.
        # Wait, the base ServingEngine is initialized with a model_id.
        # If we have multiple models, we need to decide which node serves which.
        # For this exercise, let's assume all nodes can serve all models (idealized)
        # OR we assign them round-robin.
        
        # Let's assign models round-robin to nodes for better realism if not specified
        unique_models = list(set(wl.model_id for wl in self.workloads))
        if unique_models:
            for i, (node_id, engine) in enumerate(self.serving_engines.items()):
                model = unique_models[i % len(unique_models)]
                engine.model_id = model
                # Update the config too so it matches
                for node in self.config.nodes:
                    if node.node_id == node_id:
                        node.model_id = model

    def schedule_request_arrivals(self, duration: Optional[float]) -> float:
        """Schedule request arrivals for all workloads."""
        max_arrival_time = 0.0
        
        for workload in self.workloads:
            # Use the simulation duration argument if provided (override),
            # otherwise use the workload's configured duration
            gen_duration = duration if duration is not None else workload.duration
            
            arrival_times = workload.arrival_process.generate_arrivals(
                start=self.start_time,
                duration=gen_duration
            )

            for arrival_time in arrival_times:
                max_arrival_time = max(max_arrival_time, arrival_time)
                self.request_counter += 1
                request_id = f"req_{self.request_counter}"

                input_length = sample_from_distribution(*workload.input_dist)
                output_length = sample_from_distribution(*workload.output_dist)

                arrival_event = Event(
                    timestamp=arrival_time,
                    event_type=EventType.REQUEST_ARRIVAL,
                    target="cluster_manager",
                    data={
                        'request_id': request_id,
                        'model': workload.model_id,
                        'input_length': input_length,
                        'output_length': output_length
                    },
                    priority=EventPriority.MEDIUM
                )
                self.event_loop.schedule_event(arrival_event)
                
        return max_arrival_time

def serialize_workload(workload: WorkloadConfig) -> Dict[str, Any]:
    """Serialize WorkloadConfig to a dictionary."""
    data = asdict(workload)
    # Handle ArrivalProcess which is not a dataclass
    if hasattr(workload.arrival_process, 'params'):
        rate, cv = workload.arrival_process.params()
        data['arrival_process'] = {
            'type': workload.arrival_process.__class__.__name__,
            'rate': rate,
            'cv': cv
        }
    else:
        data['arrival_process'] = str(workload.arrival_process)
    return data

def serialize_hardware(config: ClusterConfiguration) -> Dict[str, Any]:
    """Serialize ClusterConfiguration to a dictionary."""
    return asdict(config)

def run_single_simulation(args):
    """Run a single simulation with the specified parameters."""
    
    placement_metadata = {}
    if args.config:
        print(f"Loading configuration from: {args.config}")
        cluster_config, workloads, placement_metadata = load_config(args.config)
        cluster = MultiModelClusterManager(cluster_config, workloads)
    else:
        raise ValueError("No configuration file provided")
    cluster.run_simulation(
        duration=args.duration,
        enable_failures=False
    )

    results = cluster.get_results()
    
    if results['completed_requests']:
        latencies = [
            req['generation_finished_at'] - req['arrive_at']
            for req in results['completed_requests']
            if req['generation_finished_at'] is not None
        ]
        if latencies:
            p95_latency = np.percentile(latencies, 95)
            print(f"P95 Latency: {p95_latency:.3f}s")
            results['metrics']['p95_latency'] = p95_latency

    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, 'results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
        
        # Save simulation details
        details_file = os.path.join(args.output_dir, 'simulation_details.json')
        details = {
            "placement_metadata": placement_metadata,
            "metrics": results.get('metrics', {}),
            "args": vars(args),
            "hardware": serialize_hardware(cluster_config),
            "workload": [serialize_workload(wl) for wl in workloads],
        }
        with open(details_file, 'w') as f:
            json.dump(details, f, indent=2, default=str)
        print(f"Simulation details saved to: {details_file}")
        
        try:
            export_chrome_trace_from_results(results, os.path.join(args.output_dir, 'chrome_trace.json'))
            print(f"tracefile exported to: {os.path.join(args.output_dir, 'chrome_trace.json')}")
        except Exception as e:
            print(f"Error exporting Chrome trace: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run LLM serving cluster simulation")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--scheduler", type=str, default="random", choices=["random", "round_robin"], help="Scheduling algorithm to use")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds (optional, overrides workload duration)")
    parser.add_argument("--output-dir", type=str, help="Output directory for results (JSON format)")

    args = parser.parse_args()

    results = run_single_simulation(args)

    return results


if __name__ == "__main__":
    main()