from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

from simulator.core.cluster_manager import NodeConfiguration
from simulator.core.model_analyzer import ModelAnalyzer
from simulator.configs.hardware import hardware_params
from simulator.core.config import ParallelConfig, WorkloadConfig
from simulator.core.events import Event, EventType, EventPriority
import logging
import itertools
from copy import deepcopy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class PhysicalNodeConfig:
    """Configuration for a physical node."""
    gpu_type: str
    count: int
    gpus_per_node: int
    cost: float = 0.0

class PlacementStrategy(ABC):
    """Abstract base class for placement strategies."""
    
    @abstractmethod
    def place(self, physical_nodes: List[PhysicalNodeConfig], workloads: List[WorkloadConfig]) -> Tuple[List[NodeConfiguration], Dict[str, Any]]:
        """
        Determine how to split physical nodes into replicas.
        Args:
            physical_nodes: List of physical node configurations
            workloads: List of workloads to be served
        Returns:
            Tuple containing:
            - List of NodeConfiguration objects representing the logical replicas
            - Dictionary containing metadata about the placement decision (e.g., candidates evaluated)
        """
        pass

class MaximizeReplicasStrategy(PlacementStrategy):
    """
    Strategy that tries to create as many replicas as possible.
    It finds the minimum TP size required to fit the model and splits nodes accordingly.
    If multiple models are present, it distributes physical nodes among them in a round-robin fashion.
    """
    
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold

    def place(self, physical_nodes: List[PhysicalNodeConfig], workloads: List[WorkloadConfig]) -> Tuple[List[NodeConfiguration], Dict[str, Any]]:
        logical_nodes = []
        node_counter = 0
        total_cost = 0.0
        
        if not workloads:
            return [], {"strategy": "maximize_replicas", "reason": "no_workloads"}
            
        # 1. Sort physical nodes by memory capacity (ascending)
        # This ensures we use "weaker" nodes for "smaller" models first, saving "stronger" nodes for "larger" models.
        # We flatten first.
        individual_nodes = []
        for p_node in physical_nodes:
            for _ in range(p_node.count):
                individual_nodes.append(PhysicalNodeConfig(
                    gpu_type=p_node.gpu_type,
                    count=1,
                    gpus_per_node=p_node.gpus_per_node,
                    cost=p_node.cost
                ))
        
        # Sort nodes by vmemory
        individual_nodes.sort(key=lambda n: hardware_params[n.gpu_type]['vmemory'])
        
        # 2. Sort workloads by estimated memory requirement (descending)
        # This is a heuristic to favor placing larger models when replica counts are tied.
        # Let's try to analyze.
        workload_mem_estimates = {}
        for wl in workloads:
            try:
                # Quick estimate with TP=1
                # We reuse the _check_fits logic but get the value
                # Note: This might be slow if we load configs.
                # Optimization: Cache this.
                # For now, let's just use a dummy value if analysis fails, or 0.
                # We can also just use the order provided by user if we assume they know?
                # But let's try to be smart.
                
                # We'll use a helper to get memory
                mem = self._estimate_model_memory(wl.model_id)
                workload_mem_estimates[wl.model_id] = mem
            except:
                raise Exception(f"Failed to estimate memory for model {wl.model_id}")
                workload_mem_estimates[wl.model_id] = 0
            
        # Sort workloads: primary key = memory (descending)
        sorted_workloads = sorted(workloads, key=lambda w: workload_mem_estimates.get(w.model_id, 0), reverse=True)
        
        # Track replica counts
        replica_counts = {wl.model_id: 0 for wl in workloads}
        
        for p_node in individual_nodes:
            # Find all compatible workloads
            compatible_workloads = []
            
            for wl in sorted_workloads:
                if wl.tensor_parallel_size is not None:
                    min_tp = wl.tensor_parallel_size
                else:
                    min_tp = self._find_min_tp_size(p_node.gpu_type, wl.model_id, p_node.gpus_per_node)
                if min_tp <= p_node.gpus_per_node:
                    compatible_workloads.append((wl, min_tp))
            
            if not compatible_workloads:
                print(f"Warning: Node {p_node.gpu_type} cannot run any of the requested models. Skipping.")
                continue
                
            # Pick the best workload
            # Criteria:
            # 1. Minimize current replica count (balance)
            # 2. Maximize model memory (prefer large models on capable hardware) - handled by sorted_workloads order
            
            # Since sorted_workloads is already sorted by memory desc, 
            # we just need to find the one with min replica count.
            # If ties, the first one (larger memory) wins.
            
            best_wl = min(compatible_workloads, key=lambda x: replica_counts[x[0].model_id])
            logger.info(f"Selected workload: {best_wl[0].model_id} (TP={best_wl[1]})")
            selected_workload, tp_size = best_wl
            model_id = selected_workload.model_id
            
            replicas_per_node = p_node.gpus_per_node // tp_size
            
            print(f"Placement: {p_node.gpu_type} (x1) -> {replicas_per_node} replicas of {model_id} (TP={tp_size})")
            
            total_cost += p_node.cost
            
            # Update count
            replica_counts[model_id] += replicas_per_node
            
            for r in range(replicas_per_node):
                node_id = f"node_{node_counter}"
                logical_nodes.append(NodeConfiguration(
                    node_id=node_id,
                    model_id=model_id,
                    hardware=p_node.gpu_type,
                    # max_batch_size=32, # Deprecated: now managed dynamically by MemoryPlanner
                    parallel_config=ParallelConfig(tensor_parallel_size=tp_size)
                ))
                node_counter += 1
                    

        
        return logical_nodes, {
            "strategy": "maximize_replicas", 
            "replica_counts": replica_counts,
            "hourly_cost": total_cost
        }

    def _estimate_model_memory(self, model_id: str) -> float:
        """Estimate model memory usage for sorting."""
        # Reuse _check_fits logic but return value
        # Simplified: just return 0 if fails
        try:
             # Try to load appropriate config module
            config_module = None
            if "llama" in model_id.lower() or "Llama" in model_id:
                try:
                    from simulator.configs.models import llama
                    config_module = llama
                except ImportError:
                    pass
            
            if config_module is None:
                return 0

            analyzer = ModelAnalyzer(model_id, config_module, "NVDA:A100") # Hardware doesn't matter much for weight size
            results = analyzer.analyze(seqlen=1, batchsize=1, tp_size=1)
            return results['total_results']['prefill']['memory_consumption']
        except:
            return 0

    def _find_min_tp_size(self, hardware: str, model_id: str, max_tp: int) -> int:
        """Find the minimum TP size required to fit the model in memory."""
        # We need to instantiate a ModelAnalyzer to check memory usage
        # This is a bit tricky because ModelAnalyzer expects a config object sometimes
        # Let's try to use it without specific config first, relying on AutoConfig
        
        try:
            # We can't easily get the exact config object here without loading the model
            # But ModelAnalyzer loads AutoConfig internally if we pass None for config
            # However, we need to handle the case where it might fail or need specific model config
            
            # For now, let's iterate powers of 2 for TP size: 1, 2, 4, 8... up to max_tp
            tp_candidates = [2**i for i in range(int(math.log2(max_tp)) + 1)]
            
            for tp in tp_candidates:
                if tp > max_tp:
                    break
                    
                if self._check_fits(hardware, model_id, tp):
                    return tp
            
            return max_tp + 1 # Indicates failure
            
        except Exception as e:
            print(f"Error determining TP size: {e}. Defaulting to TP=1 if possible, or max_tp.")
            # Fallback: if we can't analyze, assume TP=1 works? Or maybe safe bet is max_tp?
            # Let's be optimistic for now, or maybe conservative?
            # If we fail to analyze, we might default to 1 and let it OOM at runtime?
            # Or default to max_tp to be safe?
            return 1

    def _check_fits(self, hardware: str, model_id: str, tp_size: int) -> bool:
        """Check if model fits in memory with given TP size."""
        try:
            # Try to load appropriate config module
            config_module = None
            if "llama" in model_id.lower() or "Llama" in model_id:
                try:
                    from simulator.configs.models import llama
                    config_module = llama
                except ImportError:
                    pass
            
            if config_module is None:
                logger.warning(f"Warning: No config module found for model {model_id}. Cannot analyze memory usage.")
                return False

            analyzer = ModelAnalyzer(model_id, config_module, hardware)
            
            # Analyze with a minimal batch/seqlen to get static memory footprint
            # We care about weights + KV cache reserve
            # ModelAnalyzer.analyze returns a dict with memory info
            
            results = analyzer.analyze(seqlen=1, batchsize=1, tp_size=tp_size)
            
            # Get memory consumption
            # The results structure is: results['total_results']['prefill']['memory_consumption']
            # This includes weights + KV for that run
            
            total_mem_needed = results['total_results']['prefill']['memory_consumption']
            
            # Check against hardware memory
            # Hardware params gives memory per GPU
            gpu_mem = hardware_params[hardware]['vmemory']
            
            # With TP, the model is split, so each GPU holds 1/TP of weights
            # BUT ModelAnalyzer should already account for this if we pass tp_size?
            # Let's check ModelAnalyzer code again.
            # It seems it calculates total OPs and memory access.
            # It doesn't explicitly divide weight memory by TP size in the output?
            # Wait, get_linear_layers(tp_size) is called.
            # If get_linear_layers handles splitting, then the OPs/memory returned are per-layer?
            # And then summed up.
            # If the config helper splits dimensions, then the resulting memory is for the split model?
            # Yes, likely.
            
            # So total_mem_needed is per-GPU memory usage?
            # If ModelAnalyzer simulates the workload on ONE GPU of the TP group?
            # Usually these analyzers report total stats or per-device stats.
            # Given it uses hardware_params to get bandwidth of ONE GPU,
            # it likely calculates per-GPU metrics.
            
            # Let's assume total_mem_needed is what's needed on ONE GPU.
            
            # Add a safety buffer (e.g. 10% or fixed amount)
            # And maybe reserve for KV cache?
            # The analyze call used seqlen=1, so minimal KV.
            # We should ensure there is room for actual serving.
            # Let's require at least 20% free for KV cache and overhead.
            
            return total_mem_needed < (gpu_mem * self.memory_threshold)
            
        except Exception as e:
            print(f"Error checking fit for TP={tp_size}: {e}")
            return False

class WorkloadBalancedPlacementStrategy(PlacementStrategy):
    """
    Strategy that allocates resources proportional to the estimated load.
    Load Score = Arrival Rate * (Avg Input Length + Avg Output Length)
    """
    
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold
        self.maximize_replicas = MaximizeReplicasStrategy(memory_threshold)

    def place(self, physical_nodes: List[PhysicalNodeConfig], workloads: List[WorkloadConfig]) -> Tuple[List[NodeConfiguration], Dict[str, Any]]:
        if not workloads:
            return [], {"strategy": "workload_balanced", "reason": "no_workloads"}
            
        # 1. Calculate Load Score for each workload
        load_scores = {}
        total_load = 0.0
        
        for wl in workloads:
            # Estimate average input/output length
            # Assuming distribution format is (type, [params])
            # We need a helper to get mean from distribution params
            avg_input = self._get_mean_from_dist(wl.input_dist)
            avg_output = self._get_mean_from_dist(wl.output_dist)
            
            arrival_rate = wl.arrival_process.rate()
            
            load_score = arrival_rate * (avg_input + avg_output)
            load_scores[wl.model_id] = load_score
            total_load += load_score
            
        logger.info(f"Workload Load Scores: {load_scores}")
        
        # 2. Flatten physical nodes
        individual_nodes = []
        for p_node in physical_nodes:
            for _ in range(p_node.count):
                individual_nodes.append(PhysicalNodeConfig(
                    gpu_type=p_node.gpu_type,
                    count=1,
                    gpus_per_node=p_node.gpus_per_node,
                    cost=p_node.cost
                ))
        
        # Sort nodes by vmemory (descending) to give best nodes to highest load?
        # Or just sort by capacity to ensure we fit models.
        individual_nodes.sort(key=lambda n: hardware_params[n.gpu_type]['vmemory'], reverse=True)
        
        # 3. Allocation
        # We want to assign nodes such that the aggregate compute/memory capacity 
        # assigned to a workload is proportional to its load score.
        # This is a bit complex because nodes are heterogeneous.
        # Simplified approach: Target Replicas ~ Total Nodes * (Load Score / Total Load)
        
        total_nodes = len(individual_nodes)
        target_replicas = {}
        current_replicas = {wl.model_id: 0 for wl in workloads}
        
        for wl in workloads:
            if total_load > 0:
                share = load_scores[wl.model_id] / total_load
                target = share * total_nodes
                target_replicas[wl.model_id] = target
            else:
                target_replicas[wl.model_id] = total_nodes / len(workloads)
                
        logger.info(f"Target Replicas: {target_replicas}")
        
        logical_nodes = []
        node_counter = 0
        total_cost = 0.0
        
        # We iterate through available nodes and assign them to the workload 
        # that is most "under-served" relative to its target.
        # Metric: current / target (lower is more under-served)
        
        for p_node in individual_nodes:
            # Find compatible workloads
            compatible_workloads = []
            for wl in workloads:
                if wl.tensor_parallel_size is not None:
                    min_tp = wl.tensor_parallel_size
                else:
                    min_tp = self.maximize_replicas._find_min_tp_size(p_node.gpu_type, wl.model_id, p_node.gpus_per_node)
                if min_tp <= p_node.gpus_per_node:
                    compatible_workloads.append((wl, min_tp))
            
            if not compatible_workloads:
                logger.warning(f"Node {p_node.gpu_type} cannot run any model. Skipping.")
                continue
                
            # Select workload with lowest current/target ratio
            def get_saturation_ratio(wl_tuple):
                wl = wl_tuple[0]
                target = target_replicas[wl.model_id]
                if target == 0: return float('inf')
                return current_replicas[wl.model_id] / target
                
            best_wl_tuple = min(compatible_workloads, key=get_saturation_ratio)
            selected_workload, tp_size = best_wl_tuple
            model_id = selected_workload.model_id
            
            replicas_per_node = p_node.gpus_per_node // tp_size
            
            total_cost += p_node.cost
            
            # Update count
            # We count "nodes assigned", not just replicas, for the ratio to match target_replicas (which is in units of nodes)
            # Actually target_replicas was calculated based on physical nodes count.
            # So we increment by 1 (since we are processing 1 physical node)
            current_replicas[model_id] += 1 
            
            for r in range(replicas_per_node):
                node_id = f"node_{node_counter}"
                logical_nodes.append(NodeConfiguration(
                    node_id=node_id,
                    model_id=model_id,
                    hardware=p_node.gpu_type,
                    # max_batch_size=32, # Deprecated: now managed dynamically by MemoryPlanner
                    parallel_config=ParallelConfig(tensor_parallel_size=tp_size)
                ))
                node_counter += 1
                
        return logical_nodes, {
            "strategy": "workload_balanced",
            "load_scores": load_scores,
            "target_replicas": target_replicas,
            "target_replicas": target_replicas,
            "final_replicas": current_replicas,
            "hourly_cost": total_cost
        }

    def _get_mean_from_dist(self, dist_tuple: Tuple[str, List[float]]) -> float:
        dist_type, params = dist_tuple
        if dist_type == "Poisson":
            return params[0]
        elif dist_type == "Gamma":
            # Mean = rate * shape * scale? No.
            # GammaProcess in arrival.py takes (rate, cv).
            # But here input_dist is likely (Normal, [mean, std]) or (Uniform, [min, max])
            # Based on run_cluster_simulation.py:
            # Normal: params[0] is mean
            # Uniform: (min+max)/2
            # Constant: params[0]
            pass
        
        if dist_type == "Normal":
            return params[0]
        elif dist_type == "Uniform":
            return (params[0] + params[1]) / 2
        elif dist_type == "Constant":
            return params[0]
        elif dist_type == "Poisson": # If used for length?
            return params[0]
            
        return 100.0 # Fallback


class SimulationSearchPlacementStrategy(PlacementStrategy):
    """
    Strategy that enumerates candidate placements and runs simulations to find the best one.
    """
    
    def __init__(self, memory_threshold: float = 0.8, simulation_duration: float = 10.0):
        self.memory_threshold = memory_threshold
        self.simulation_duration = simulation_duration
        self.maximize_replicas = MaximizeReplicasStrategy(memory_threshold)

    def place(self, physical_nodes: List[PhysicalNodeConfig], workloads: List[WorkloadConfig]) -> Tuple[List[NodeConfiguration], Dict[str, Any]]:
        # Avoid circular imports
        from simulator.core.cluster_manager import ClusterManager, ClusterConfiguration
        
        if not workloads:
            return [], {"strategy": "simulation_search", "reason": "no_workloads"}
            
        # 1. Group identical physical nodes
        # We want to partition these groups among workloads.
        # Example: 2x A100, 4x 3090.
        # Workloads: A, B.
        # A100 partition: (2,0), (1,1), (0,2)
        # 3090 partition: (4,0), (3,1), ... (0,4)
        # Total combinations: 3 * 5 = 15.
        
        node_groups = {} # key: gpu_type, value: count
        for p_node in physical_nodes:
            node_groups[p_node.gpu_type] = node_groups.get(p_node.gpu_type, 0) + p_node.count
            
        gpu_types = list(node_groups.keys())
        workload_indices = range(len(workloads))
        
        # Generate partitions for each GPU type
        type_partitions = []
        for gpu_type in gpu_types:
            count = node_groups[gpu_type]
            # Generate all ways to split 'count' items into 'len(workloads)' bins
            # Stars and bars: (n+k-1) choose (k-1)
            # We can use itertools.product if we iterate counts? No.
            # We need partitions of integer n into k parts.
            
            partitions = []
            # Simple recursive generator
            def generate_partitions(n, k):
                if k == 1:
                    yield (n,)
                    return
                for i in range(n + 1):
                    for p in generate_partitions(n - i, k - 1):
                        yield (i,) + p
                        
            partitions = list(generate_partitions(count, len(workloads)))
            type_partitions.append(partitions)
            
        # Cartesian product of partitions for all types
        all_combinations = list(itertools.product(*type_partitions))
        
        best_config = []
        best_throughput = -1.0
        candidates_metadata = []
        
        logger.info(f"Evaluating {len(all_combinations)} placement candidates...")
        
        for i, combination in enumerate(all_combinations):
            # combination is a tuple of partitions, one for each GPU type
            # e.g. ( (2,0), (1,3) ) for 2 types and 2 workloads
            
            # Construct the assignment
            # workload_assignments[wl_idx] = list of (gpu_type, count)
            workload_assignments = [[] for _ in workloads]
            
            valid_assignment = True
            
            for type_idx, partition in enumerate(combination):
                gpu_type = gpu_types[type_idx]
                # Find the physical node config to get cost and gpus_per_node
                p_node_ref = next(n for n in physical_nodes if n.gpu_type == gpu_type)
                gpus_per_node = p_node_ref.gpus_per_node
                node_cost = p_node_ref.cost
                
                for wl_idx, count in enumerate(partition):
                    if count > 0:
                        # Check if model fits
                        model_id = workloads[wl_idx].model_id
                        if workloads[wl_idx].tensor_parallel_size is not None:
                            min_tp = workloads[wl_idx].tensor_parallel_size
                        else:
                            min_tp = self.maximize_replicas._find_min_tp_size(gpu_type, model_id, gpus_per_node)
                        
                        if min_tp > gpus_per_node:
                            valid_assignment = False
                            break
                            
                        workload_assignments[wl_idx].append({
                            'gpu_type': gpu_type,
                            'count': count,
                            'gpus_per_node': gpus_per_node,
                            'tp_size': min_tp,
                            'cost': node_cost
                        })
                if not valid_assignment:
                    break
            
            if not valid_assignment:
                continue
                
            # Construct NodeConfigurations
            candidate_nodes = []
            node_counter = 0
            candidate_cost = 0.0
            
            for wl_idx, assignments in enumerate(workload_assignments):
                model_id = workloads[wl_idx].model_id
                for assign in assignments:
                    gpu_type = assign['gpu_type']
                    count = assign['count']
                    tp_size = assign['tp_size']
                    gpus_per_node = assign['gpus_per_node']
                    cost = assign['cost']
                    
                    candidate_cost += count * cost
                    
                    # For each physical node assigned
                    for _ in range(count):
                        replicas = gpus_per_node // tp_size
                        for _ in range(replicas):
                            candidate_nodes.append(NodeConfiguration(
                                node_id=f"node_{node_counter}",
                                model_id=model_id,
                                hardware=gpu_type,
                                # max_batch_size=32, # Deprecated
                                parallel_config=ParallelConfig(tensor_parallel_size=tp_size)
                            ))
                            node_counter += 1
                            
            if not candidate_nodes:
                continue
                
            # Run Simulation
            try:
                # Create a temporary cluster config
                temp_config = ClusterConfiguration(
                    cluster_id=f"sim_search_{i}",
                    nodes=candidate_nodes,
                    scheduler_algorithm="round_robin" # Use simple scheduler for evaluation
                )
                
                # We need a custom ClusterManager that schedules arrivals for all workloads.
                class SimClusterManager(ClusterManager):
                    def __init__(self, config, workloads):
                        self.workloads = workloads
                        super().__init__(config, workloads[0].arrival_process)
                        
                    def schedule_request_arrivals(self, duration):
                        for wl in self.workloads:
                            times = wl.arrival_process.generate_arrivals(self.start_time, duration)
                            for t in times:
                                self.request_counter += 1
                                # Simple sampling
                                def sample(dist):
                                    dtype, params = dist
                                    if dtype == "Normal": return max(1, int(np.random.normal(params[0], params[1])))
                                    if dtype == "Uniform": return max(1, int(np.random.uniform(params[0], params[1])))
                                    if dtype == "Constant": return int(params[0])
                                    return 100
                                
                                inp = sample(wl.input_dist)
                                out = sample(wl.output_dist)
                                
                                event = Event(
                                    timestamp=t,
                                    event_type=EventType.REQUEST_ARRIVAL,
                                    target="cluster_manager",
                                    data={
                                        'request_id': f"req_{self.request_counter}",
                                        'model': wl.model_id,
                                        'input_length': inp,
                                        'output_length': out
                                    },
                                    priority=EventPriority.MEDIUM
                                )
                                self.event_loop.schedule_event(event)
                                
                # Run simulation
                import numpy as np
                
                sim_manager = SimClusterManager(temp_config, workloads)
                sim_manager.run_simulation(self.simulation_duration, enable_failures=False)
                
                throughput = sim_manager.metrics['throughput']
                
                logger.info(f"Candidate {i}: Throughput={throughput:.2f} req/s")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = candidate_nodes
                
                candidates_metadata.append({
                    "candidate_id": i,
                    "throughput": throughput,
                    "hourly_cost": candidate_cost,
                    "assignment": [
                        {
                            "workload": workloads[idx].model_id,
                            "gpu_type": assign['gpu_type'],
                            "count": assign['count'],
                            "tp_size": assign['tp_size']
                        }
                        for idx, assignments in enumerate(workload_assignments)
                        for assign in assignments
                    ]
                })
                    
            except Exception as e:
                logger.error(f"Simulation failed for candidate {i}: {e}")
                
        return best_config, {
            "strategy": "simulation_search",
            "best_throughput": best_throughput,
            "candidates": candidates_metadata
        }

class PlacementDecisionMaker:
    """Factory for creating placement decisions."""
    
    def __init__(self, strategy_name: str = "maximize_replicas", memory_threshold: float = 0.8):
        self.strategy = self._get_strategy(strategy_name, memory_threshold)
        
    def _get_strategy(self, name: str, memory_threshold: float) -> PlacementStrategy:
        if name == "maximize_replicas":
            return MaximizeReplicasStrategy(memory_threshold)
        elif name == "workload_balanced":
            return WorkloadBalancedPlacementStrategy(memory_threshold)
        elif name == "simulation_search":
            return SimulationSearchPlacementStrategy(memory_threshold)
        else:
            raise ValueError(f"Unknown placement strategy: {name}")
            
    def place(self, physical_nodes: List[PhysicalNodeConfig], workloads: List[WorkloadConfig]) -> Tuple[List[NodeConfiguration], Dict[str, Any]]:
        return self.strategy.place(physical_nodes, workloads)
