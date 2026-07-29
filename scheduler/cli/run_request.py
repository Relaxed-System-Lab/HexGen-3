#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
import random
import numpy as np
import httpx
import math
from typing import Optional
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadGenerator:
    def __init__(
        self,
        arrival_rate: float,
        input_len: int,
        output_len: int,
        endpoint: str,
        model_name: str,
        duration: Optional[int] = None
    ):
        self.arrival_rate = arrival_rate
        self.input_len = input_len
        self.output_len = output_len
        self.endpoint = endpoint
        self.duration = duration
        self.model_name = model_name
        
        logger.info(f"Loading tokenizer for model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = self._generate_prompt()
        
        self.client = httpx.AsyncClient(timeout=120.0)
        self.stats = {
            "requests": 0,
            "success": 0,
            "errors": 0,
            "latencies": []
        }

        self.tasks = set()

    def _generate_prompt(self) -> str:
        """Generate a prompt with exact token length."""
        # Start with a simple repeated word
        word = "test "
        # Estimate needed repetitions (rough guess)
        estimated_chars = self.input_len * 4
        text = word * (estimated_chars // len(word))
        
        # Tokenize and adjust
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < self.input_len:
            # Add more
            while len(tokens) < self.input_len:
                text += " test"
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate to exact length
        if len(tokens) > self.input_len:
            tokens = tokens[:self.input_len]
            text = self.tokenizer.decode(tokens)
        logger.info(f"Generated prompt with {len(tokens)} tokens")
        return text

    async def send_request(self, request_id: int):
        """Send a single request."""
        payload = {
            "model": self.model_name,
            "prompt": self.prompt,
            "max_tokens": self.output_len,
            "stream": False
        }
        
        start_time = time.time()
        try:
            logger.debug(f"Sending request {request_id}")
            response = await self.client.post(self.endpoint, json=payload)
            latency = time.time() - start_time
            
            self.stats["requests"] += 1
            self.stats["latencies"].append(latency)
            
            if response.status_code == 200:
                self.stats["success"] += 1
                logger.info(f"Request {request_id} success, latency: {latency:.4f}s")
            else:
                self.stats["errors"] += 1
                logger.error(f"Request {request_id} failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.stats["requests"] += 1
            self.stats["errors"] += 1
            logger.error(f"Request {request_id} error: {e}")

    async def run(self):
        """Run the load generator."""
        logger.info(f"Starting load generator at {self.arrival_rate} req/s")
        start_time = time.time()
        request_id = 0
        
        try:
            while True:
                if self.duration and (time.time() - start_time) > self.duration:
                    break
                
                # Calculate next arrival time (Poisson process)
                # Inter-arrival time = -math.log(random.random()) / self.arrival_rate
                inter_arrival = -math.log(random.random()) / self.arrival_rate
                
                await asyncio.sleep(inter_arrival)
                
                # Fire and forget (create task)
                task = asyncio.create_task(self.send_request(request_id))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                request_id += 1
                
        except KeyboardInterrupt:
            logger.info("Stopping load generator...")
        finally:
            # Wait for pending requests
            if self.tasks:
                logger.info(f"Waiting for {len(self.tasks)} pending requests...")
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self._print_stats()
            await self.client.aclose()

    def _print_stats(self):
        """Print final statistics."""
        latencies = sorted(self.stats["latencies"])
        if not latencies:
            logger.info("No requests completed.")
            return
            
        avg_lat = sum(latencies) / len(latencies)
        
        def get_percentile(data, p):
            k = (len(data) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d0 = data[int(f)]
            d1 = data[int(c)]
            return d0 + (d1 - d0) * (k - f)

        p50 = get_percentile(latencies, 50)
        p99 = get_percentile(latencies, 99)
        
        print("\n" + "="*40)
        print(f"Total Requests: {self.stats['requests']}")
        print(f"Success: {self.stats['success']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Average Latency: {avg_lat:.4f}s")
        print(f"P50 Latency: {p50:.4f}s")
        print(f"P99 Latency: {p99:.4f}s")
        print("="*40 + "\n")

async def main():
    parser = argparse.ArgumentParser(description="LLM Load Generator")
    parser.add_argument("--arrival-rate", type=float, required=True, help="Requests per second")
    parser.add_argument("--input-len", type=int, required=True, help="Input prompt length in tokens")
    parser.add_argument("--output-len", type=int, required=True, help="Output length in tokens")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/completions", help="Target endpoint URL")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model name (used for tokenizer and request)")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    
    args = parser.parse_args()
    
    generator = LoadGenerator(
        arrival_rate=args.arrival_rate,
        input_len=args.input_len,
        output_len=args.output_len,
        endpoint=args.endpoint,
        model_name=args.model,
        duration=args.duration
    )
    
    await generator.run()

if __name__ == "__main__":
    asyncio.run(main())
