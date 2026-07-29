import logging
import os
from cluster_experiments.common.spinup import SpinUpManager

logging.basicConfig(level=logging.INFO)

def test_spinup():
    print("Testing SpinUpManager...")
    
    # Mock config
    config = {
        "template": "cluster_experiments/exp1_scaling/scaling.jinja",
        "model": "meta-llama/Llama-2-7b-hf",
        "port": 8000,
        "tensor_parallel_size": 1,
        "nodes": 2
    }
    
    # Initialize
    manager = SpinUpManager(config, template_path=config["template"])
    
    # Test rendering
    print("\n--- Rendered Template ---")
    rendered = manager.render_template()
    print(rendered)
    print("-------------------------")
    
    # We can't actually submit to Slurm here without sbatch, 
    # but we can verify the template renders correctly.
    
    if "vllm serve meta-llama/Llama-2-7b-hf" in rendered:
        print("\nSUCCESS: Template rendered correctly.")
    else:
        print("\nFAILURE: Template did not render correctly.")

if __name__ == "__main__":
    test_spinup()
