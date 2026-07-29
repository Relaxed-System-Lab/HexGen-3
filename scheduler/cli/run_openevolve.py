import os
import random
import inspect
import asyncio
from typing import List, Tuple, Any, Dict
from openevolve import run_evolution
from openevolve.config import Config, LLMModelConfig
from simulator.core.request import GenerationRequest
from simulator.core.model_analyzer import ModelAnalyzer
from simulator.configs.hardware import hardware_params
from simulator.configs.models import llama

# Mock classes to simulate environment without full engine overhead
class MockEngine:
    def __init__(self, engine_id: str, hardware: str):
        self.engine_id = engine_id
        self.hardware = hardware
        self.current_load = 0
        
    def get_current_load(self):
        return self.current_load
        
    def __repr__(self):
        return f"Engine({self.engine_id}, {self.hardware})"

def generate_test_cases(num_cases: int = 50) -> List[Tuple[GenerationRequest, List[MockEngine]]]:
    """
    Generate test cases for evaluation.
    Returns list of (request, available_engines)
    """
    test_cases = []
    
    # Hardware profiles available in simulator
    hardware_types = ["NVDA:H100:PCIe", "NVDA:A100_80G:SXM", "NVDA:RTX3090"]
    
    for i in range(num_cases):
        # 1. Generate random request
        # Use a real model ID that ModelAnalyzer supports
        req = GenerationRequest(
            req_id=f"req_{i}",
            model="meta-llama/Llama-3.1-8B-Instruct",
            input_length=random.randint(128, 4096),
            output_length=random.randint(128, 1024),
            arrive_at=0.0
        )
        
        # 2. Generate random available engines (3-5 engines)
        engines = []
        num_engines = random.randint(3, 5)
        for j in range(num_engines):
            hw_name = random.choice(hardware_types)
            engine = MockEngine(
                engine_id=f"node_{j}",
                hardware=hw_name
            )
            # Randomize load slightly
            engine.current_load = random.randint(0, 5)
            engines.append(engine)
            
        test_cases.append((req, engines))
            
    return test_cases

class SimulatorEvaluator:
    def __init__(self, test_cases: List[Tuple[GenerationRequest, List[MockEngine]]]):
        self.test_cases = test_cases
        self.analyzers: Dict[str, ModelAnalyzer] = {}
        self._init_analyzers()
        
    def _init_analyzers(self):
        """Pre-load analyzers for each hardware type to save time."""
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        hardware_types = ["NVDA:H100:PCIe", "NVDA:A100_80G:SXM", "NVDA:RTX3090"]
        
        print("Initializing ModelAnalyzers...")
        for hw in hardware_types:
            try:
                self.analyzers[hw] = ModelAnalyzer(
                    model_id=model_id,
                    config=llama,
                    hardware=hw
                )
            except Exception as e:
                print(f"Failed to init analyzer for {hw}: {e}")

    def evaluate(self, program_path: str) -> Dict[str, Any]:
        """
        Evaluate the evolved program.
        Score is negative total latency (we want to minimize latency).
        """
        import importlib.util
        
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved", program_path)
        if spec is None or spec.loader is None:
            return {"score": -float('inf'), "error": "Failed to load program"}
        
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            return {"score": -float('inf'), "error": f"Failed to execute program: {str(e)}"}
        
        if not hasattr(module, "scheduling_algorithm"):
            return {"score": -float('inf'), "error": "Function 'scheduling_algorithm' not found"}
        
        func = getattr(module, "scheduling_algorithm")
        
        total_latency = 0.0
        errors = []
        valid_decisions = 0
        
        for req, engines in self.test_cases:
            try:
                # Call the evolved function
                # It expects a single argument (inputs) which unpacks to (request, engines)
                # But our test_cases list has (req, engines) tuples.
                # So we pass the tuple directly.
                target_engine_id = func((req, engines))
                
                # Find the engine object
                target_engine = next((e for e in engines if e.engine_id == target_engine_id), None)
                
                if target_engine:
                    # Calculate latency using ModelAnalyzer
                    analyzer = self.analyzers.get(target_engine.hardware)
                    if analyzer:
                        # Estimate latency
                        # We use batchsize=1 + current_load (simplified)
                        batch_size = 1 + target_engine.current_load
                        
                        metrics = analyzer.analyze_generate_task(
                            prompt_len=req.input_length,
                            gen_len=req.output_length,
                            batchsize=batch_size
                        )
                        
                        latency = metrics["inference_time"]
                        total_latency += latency
                        valid_decisions += 1
                    else:
                        errors.append(f"No analyzer for {target_engine.hardware}")
                        total_latency += 1000.0 # Penalty
                else:
                    # Invalid decision (None or unknown ID)
                    total_latency += 1000.0 # Penalty
                    
            except Exception as e:
                errors.append(f"Runtime error: {str(e)}")
                total_latency += 1000.0 # Penalty
        
        # Normalize score
        # If all valid, score is negative total latency
        # We want to maximize score, so minimizing latency
        
        return {
            "score": -total_latency,
            "total_latency": total_latency,
            "valid_decisions": valid_decisions,
            "errors": errors[:3]
        }

# The function to be evolved
def scheduling_algorithm(inputs):
    # Unpack inputs
    request, available_engines = inputs
    
    # EVOLVE-BLOCK-START
    # Initial simple logic: Random selection
    # The goal is to evolve this to minimize latency
    import random
    if not available_engines:
        return None
    
    # Simple baseline: pick random
    selected = available_engines[0]
    
    return selected.engine_id
    # EVOLVE-BLOCK-END

if __name__ == "__main__":
    print("Generating test cases...")
    cases = generate_test_cases(20) # 20 cases for faster eval
    print(f"Generated {len(cases)} test cases.")
    
    evaluator = SimulatorEvaluator(cases)
    
    # Get initial source code
    source_code = inspect.getsource(scheduling_algorithm)
    
    # Configure openevolve
    config = Config()
    config.llm.models = [
        LLMModelConfig(
            name="Qwen/Qwen3-Next-80B-A3B-Instruct",
            api_key=os.environ.get("OPENAI_LIKE_API_KEY"),
            api_base=os.environ.get("OPENAI_LIKE_API_BASE")
        )
    ]
    
    print("Starting evolution with custom evaluator...")
    # We use run_evolution directly for custom evaluator
    result = run_evolution(
        initial_program=source_code,
        evaluator=evaluator.evaluate,
        iterations=10, 
        config=config
    )
    
    print(f"\nBest Score: {result.best_score}")
    print(f"Best Code:\n{result.best_code}")