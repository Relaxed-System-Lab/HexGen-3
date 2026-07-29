import os
import time
import json
from typing import Optional

from simulator.utils.engine import (
    benchmark_openai_compatible_server_with_stats,
    start_vllm,
    stop_vllm,
)
from simulator.utils.gpu import get_gpu_spec

def run_real_bench(args):
    print(f"args: {args}")
    pid: Optional[int] = None
    stdout_log: Optional[str] = None
    gpu_spec = get_gpu_spec()
    try:
        pid, stdout_log = start_vllm(args.model_id)
        print(f"Started vllm with pid {pid} (log -> {stdout_log})")

        results = benchmark_openai_compatible_server_with_stats(
            model_id=args.model_id,
            input_prompt_len=args.input_length,
            output_len=args.output_length,
            base_url="http://localhost:8080/v1",
            api_key=None,
            request_timeout=args.timeout,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
        )

        # Print summary results
        summary = results["summary"]
        print(f"\n=== Benchmark Results ===")
        print(f"Model: {results['model_id']}")
        print(f"Input length: {results['input_prompt_len']} tokens")
        print(f"Output length: {results['output_len']} tokens")
        print(f"Iterations completed: {results['iterations_completed']}")
        print(
            f"Prefill: {summary['avg_prefill_time_ms']:.2f}±{summary['std_prefill_time_ms']:.2f} ms"
        )
        print(
            f"Decode: {summary['avg_decode_time_ms_per_token']:.3f}±{summary['std_decode_time_ms_per_token']:.3f} ms/token"
        )
        print(f"Throughput: {summary['tokens_per_second']:.1f} tokens/sec")

        # Save detailed results to JSON file if requested
        if args.output_dir:
            # hostname + timestamp
            hostname = os.uname().nodename
            timestamp = int(time.time())
            output_file = os.path.join(args.output_dir, f"{hostname}_{timestamp}_benchmark_results.json")
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "gpu_spec": gpu_spec,
                        "benchmark_results": results,
                        "server_config": {
                            "model_id": args.model_id,
                            "tp_size": 1,
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"Detailed results saved to: {args.output_file}")

    finally:
        if pid is not None:
            try:
                stop_vllm(pid)
                print(f"Stopped vllm process {pid}")
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                print(f"Warning: failed to stop vllm process {pid}: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM serving benchmark with statistics"
    )
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of benchmark iterations to run",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=3,
        help="Number of warmup iterations to discard from statistics",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="dir to save detailed results",
    )
    run_real_bench(parser.parse_args())
