"""
Run few-shot GSM-8K evaluation for PD (Prefill-Decode) disaggregation mode.

Usage:
python3 -m sglang.test.few_shot_gsm8k_pd --num-questions 200 --router-url http://127.0.0.1:30000

Note: The router will automatically:
1. Select prefill and decode server pairs
2. Inject bootstrap info (bootstrap_host, bootstrap_port, bootstrap_room)
3. Forward requests to both servers
"""

import argparse
import ast
import re
import time
from typing import List, Dict, Any

import numpy as np
import requests

from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def send_pd_request(
    router_url: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Send a request to the router in PD disaggregation mode.
    
    The router will:
    1. Select prefill and decode server pair
    2. Inject bootstrap info (bootstrap_host, bootstrap_port, bootstrap_room)
    3. Forward the request to both prefill and decode servers
    4. Return the decode server's response (final result)
    
    Args:
        router_url: URL of the router server (e.g., "http://127.0.0.1:30000")
        prompt: The prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds
    
    Returns:
        Response JSON from the router (which contains decode server's response)
    """
    request_data = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": ["Question", "Assistant:", "<|separator|>"],
        },
        # Note: bootstrap_host, bootstrap_port, bootstrap_room are automatically
        # injected by the router - we don't need to specify them here
    }
    
    router_url_full = f"{router_url}/generate"
    print(f"[DEBUG] Sending POST to router={router_url_full}, prompt_len={len(prompt)}")
    start_time = time.perf_counter()
    
    try:
        response = requests.post(router_url_full, json=request_data, timeout=timeout)
        elapsed = time.perf_counter() - start_time
        print(f"[DEBUG] Router responded in {elapsed:.2f}s, status={response.status_code}")
        
        if response.status_code != 200:
            print(f"[ERROR] Router response status {response.status_code}, text: {response.text[:200]}")
            raise RuntimeError(
                f"Router request failed with status {response.status_code}: {response.text}"
            )
        
        result = response.json()
        print(f"[DEBUG] Response parsed successfully, keys: {list(result.keys())}")
        return result
        
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start_time
        print(f"[ERROR] Request timed out after {elapsed:.2f}s (timeout={timeout}s)")
        raise
    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start_time
        print(f"[ERROR] Request exception after {elapsed:.2f}s: {type(e).__name__}: {e}")
        raise


def run_eval_pd(args):
    """
    Run GSM-8K evaluation in PD disaggregation mode.
    """
    if args.data_path is None:
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.data_path

    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    # Prepare prompts with few-shot examples
    prompts = [few_shot_examples + q for q in questions]

    # Run requests
    print(f"Sending {num_questions} requests to router at {args.router_url}...")
    print(f"Router will handle server selection and bootstrap info injection")
    
    tic = time.perf_counter()
    results = []
    errors = []
    
    # Send requests in parallel batches
    import concurrent.futures
    
    def send_single_request(i, prompt):
        start_time = time.perf_counter()
        try:
            print(f"[DEBUG] Request {i}: Sending request to router")
            result = send_pd_request(
                router_url=args.router_url,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - start_time
            print(f"[DEBUG] Request {i}: Completed in {elapsed:.2f}s, response keys: {list(result.keys()) if result else 'None'}")
            return i, result, None
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"[DEBUG] Request {i}: Failed after {elapsed:.2f}s - {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return i, None, str(e)
    
    # Use ThreadPoolExecutor for parallel requests
    print(f"[DEBUG] Submitting {num_questions} requests with {args.parallel} parallel workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(send_single_request, i, prompt)
            for i, prompt in enumerate(prompts)
        ]
        print(f"[DEBUG] All {len(futures)} requests submitted, waiting for results...")
        
        # Collect results with progress bar
        completed = 0
        last_progress_time = time.perf_counter()
        for future in concurrent.futures.as_completed(futures):
            try:
                i, result, error = future.result(timeout=args.timeout + 10)  # Extra timeout for result retrieval
                if error:
                    errors.append((i, error))
                    print(f"[ERROR] Request {i} failed: {error}")
                else:
                    results.append((i, result))
                completed += 1
                current_time = time.perf_counter()
                if completed % 10 == 0 or (current_time - last_progress_time) > 5:
                    print(f"[PROGRESS] {completed}/{num_questions} requests completed ({(completed/num_questions)*100:.1f}%)")
                    last_progress_time = current_time
            except concurrent.futures.TimeoutError:
                print(f"[ERROR] Request timed out while waiting for result")
                errors.append((None, "Timeout waiting for result"))
            except Exception as e:
                print(f"[ERROR] Unexpected error collecting result: {type(e).__name__}: {e}")
                errors.append((None, str(e)))
    
    latency = time.perf_counter() - tic
    
    # Sort results by index
    results.sort(key=lambda x: x[0])
    
    # Extract predictions
    preds = []
    for i, result in results:
        if result and "text" in result:
            # Extract the answer part (after "Answer:")
            answer_text = result["text"]
            # Find the answer part after the prompt
            if "Answer:" in answer_text:
                answer_part = answer_text.split("Answer:")[-1].strip()
            else:
                answer_part = answer_text
            preds.append(get_answer_value(answer_part))
        else:
            preds.append(INVALID)
    
    # Fill in INVALID for failed requests
    failed_indices = {i for i, _ in errors}
    for i in range(num_questions):
        if i in failed_indices and len(preds) <= i:
            preds.insert(i, INVALID)
    
    # Ensure preds has the same length as labels
    while len(preds) < num_questions:
        preds.append(INVALID)
    preds = preds[:num_questions]

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    total_output_tokens = 0
    for i, result in results:
        if result and "meta_info" in result:
            total_output_tokens += result["meta_info"].get("completion_tokens", 0)
    
    if latency > 0:
        output_throughput = total_output_tokens / latency
    else:
        output_throughput = 0.0

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Failed requests: {len(errors)}/{num_questions}")
    print(f"Total latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Average latency per request: {latency/num_questions:.3f} s")
    if len(errors) > 0:
        print(f"\nErrors:")
        for i, error in errors[:10]:  # Show first 10 errors
            print(f"  Request {i}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Dump results
    if args.output_file:
        with open(args.output_file, "w") as f:
            for i, result in results:
                if result:
                    f.write(f"Question {i}:\n")
                    f.write(f"Prompt: {prompts[i][:100]}...\n")
                    f.write(f"Response: {result.get('text', 'N/A')}\n")
                    f.write(f"Predicted: {preds[i]}, Label: {labels[i]}\n")
                    f.write("-" * 60 + "\n")

    return {
        "accuracy": acc,
        "invalid": invalid,
        "latency": latency,
        "output_throughput": output_throughput,
        "failed_requests": len(errors),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GSM-8K evaluation in PD disaggregation mode"
    )
    parser.add_argument("--num-shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--data-path", type=str, default=None, help="Path to test data JSONL file")
    parser.add_argument("--num-questions", type=int, default=200, help="Number of questions to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--parallel", type=int, default=32, help="Number of parallel requests")
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="URL of the router server",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--output-file", type=str, default=None, help="Output file to save results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PD Disaggregation GSM-8K Evaluation")
    print("=" * 60)
    print(f"Router URL: {args.router_url}")
    print(f"Number of questions: {args.num_questions}")
    print(f"Parallel requests: {args.parallel}")
    print("=" * 60)
    
    # Check if router is ready
    try:
        health_response = requests.get(f"{args.router_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Router health check failed: {health_response.status_code}")
            exit(1)
        print("✅ Router is ready")
    except Exception as e:
        print(f"❌ Cannot connect to router: {e}")
        print(f"   Make sure router is running on {args.router_url}")
        print(f"   Router should be configured with prefill and decode servers")
        exit(1)
    
    results = run_eval_pd(args)
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Output Throughput: {results['output_throughput']:.3f} token/s")
    print("=" * 60)
