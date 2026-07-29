#!/usr/bin/env python3
"""Replay a decode-focused workload trace against an SGLang /generate endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_DATASET_SOURCES = [
    "WildGPT=allenai/WildChat-1M",
    "OpenThoughts=open-thoughts/OpenThoughts-114k",
    "OpenR1-Math=open-r1/OpenR1-Math-220k",
    "NuminaMath=AI-MO/NuminaMath-CoT",
]

DEFAULT_TEXT_FIELDS = (
    "prompt",
    "question",
    "problem",
    "instruction",
    "input",
    "query",
    "text",
    "messages",
    "conversation",
    "conversations",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a JSON workload trace against an SGLang server"
    )
    parser.add_argument("--trace", required=True, help="Trace JSON path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--endpoint", default="/generate")
    parser.add_argument("--output", default="/tmp/hexgen3_trace_results.jsonl")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--arrival-process",
        choices=("deterministic", "poisson"),
        default="deterministic",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-concurrency", type=int, default=512)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--http-connection-limit",
        type=int,
        default=0,
        help=(
            "aiohttp total connector limit. 0 means unlimited; this avoids the "
            "aiohttp default limit=100 from capping benchmark concurrency."
        ),
    )
    parser.add_argument(
        "--http-connection-limit-per-host",
        type=int,
        default=0,
        help="aiohttp per-host connector limit. 0 means unlimited.",
    )
    parser.add_argument("--token-low", type=int, default=10)
    parser.add_argument("--token-high", type=int, default=10000)
    parser.add_argument(
        "--prompt-source",
        choices=("random-ids", "datasets"),
        default="random-ids",
        help=(
            "Prompt source. random-ids preserves the old synthetic token-id mode; "
            "datasets samples text from dataset sources and tokenizes it."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path/name used when --prompt-source=datasets.",
    )
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=None,
        help=(
            "Dataset source for --prompt-source=datasets. Can be repeated. Format: "
            "LABEL=SOURCE[,config=CONFIG][,split=SPLIT]. SOURCE may be a HuggingFace "
            "dataset name or a local json/jsonl file. Defaults to WildGPT, "
            "OpenThoughts, OpenR1-Math, and NuminaMath candidate HF sources."
        ),
    )
    parser.add_argument(
        "--dataset-samples-per-source",
        type=int,
        default=2048,
        help="Maximum examples to materialize from each dataset source.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Optional HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--dataset-text-fields",
        default=",".join(DEFAULT_TEXT_FIELDS),
        help="Comma-separated candidate fields used to extract prompt text.",
    )
    parser.add_argument(
        "--dataset-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace load_dataset.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument(
        "--no-ignore-eos",
        action="store_false",
        dest="ignore_eos",
        help="Allow EOS before requested output length.",
    )
    parser.add_argument(
        "--max-requests-per-window",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=30.0,
        help="Print replay progress every N seconds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--progress-heartbeat-s",
        type=float,
        default=120.0,
        help=(
            "Print a progress heartbeat after this many seconds even if counters "
            "do not change. Set <=0 to disable unchanged heartbeats."
        ),
    )
    parser.add_argument(
        "--drain-state-file",
        default=None,
        help=(
            "Optional runtime state JSON file. When set, intervals with "
            "phase='reconfiguring' are excluded from per-window active throughput "
            "metrics, but still included in total wall-clock metrics. Older "
            "state files without phase fall back to excluding state='draining'."
        ),
    )
    parser.add_argument(
        "--drain-state-sample-interval-s",
        type=float,
        default=0.5,
        help="Polling interval for --drain-state-file.",
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    try:
        import aiohttp  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "missing dependency: aiohttp. Install scheduler requirements or run in the "
            "runtime environment before replaying the workload trace."
        ) from exc

    rng = random.Random(args.seed)
    windows = _load_trace(args.trace)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else None
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"http://{args.host}:{args.port}{_normalize_endpoint(args.endpoint)}"
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    all_results: List[Dict[str, object]] = []
    benchmark_start = time.time()
    total_planned = sum(_planned_requests(args, window) for window in windows)
    overall_progress = {"sent": 0, "completed": 0, "failed": 0}
    drain_recorder = DrainStateRecorder(
        args.drain_state_file,
        sample_interval_s=float(args.drain_state_sample_interval_s),
    )
    prompt_sampler = PromptSampler(args, rng)

    _print_replay_header(
        args=args,
        url=url,
        output_path=output_path,
        prompt_sampler=prompt_sampler,
        total_planned=total_planned,
    )

    timeout = aiohttp.ClientTimeout(total=max(1.0, args.request_timeout_s))
    connector = _make_http_connector(args, aiohttp)
    monitor_task = asyncio.create_task(drain_recorder.run()) if drain_recorder.enabled else None
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            with open(output_path, "w", encoding="utf-8") as out:
                for window_index, window in enumerate(windows):
                    window_results = await _run_window(
                        args=args,
                        session=session,
                        url=url,
                        window=window,
                        window_index=window_index,
                        benchmark_start=benchmark_start,
                        rng=rng,
                        prompt_sampler=prompt_sampler,
                        semaphore=semaphore,
                        out=out,
                        total_planned=total_planned,
                        overall_progress=overall_progress,
                    )
                    all_results.extend(window_results)
                    _print_window_summary(
                        window,
                        window_results,
                        drain_intervals=drain_recorder.snapshot(time.time()),
                    )
    finally:
        drain_recorder.stop()
        if monitor_task is not None:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    benchmark_end = time.time()
    drain_recorder.close(benchmark_end)
    summary = _summarize(
        windows,
        all_results,
        total_elapsed_s=benchmark_end - benchmark_start,
        benchmark_start=benchmark_start,
        drain_intervals=drain_recorder.intervals,
    )
    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"summary: {summary_path}")
    _print_overall_summary(summary)
    return 0


def _print_replay_header(
    *,
    args,
    url: str,
    output_path: Path,
    prompt_sampler: "PromptSampler",
    total_planned: int,
) -> None:
    print("HexGen-3 workload trace replay")
    print("=" * 100)
    print(f"url: {url}")
    print(f"trace: {args.trace}")
    print(f"output: {output_path}")
    print(f"arrival_process: {args.arrival_process}")
    print(f"prompt_source: {args.prompt_source}")
    if args.prompt_source == "datasets":
        print(f"dataset_prompts: {prompt_sampler.size}")
    print(f"max_concurrency: {args.max_concurrency}")
    print(f"planned_requests: {total_planned}")
    print(
        "http_connection_limit: "
        f"{args.http_connection_limit} "
        f"per_host={args.http_connection_limit_per_host}"
    )
    print("-" * 100)


def _make_http_connector(args, aiohttp):
    return aiohttp.TCPConnector(
        limit=max(0, int(args.http_connection_limit)),
        limit_per_host=max(0, int(args.http_connection_limit_per_host)),
        force_close=False,
        enable_cleanup_closed=True,
    )


class DrainStateRecorder:
    def __init__(self, path: Optional[str], *, sample_interval_s: float):
        self.path = Path(path) if path else None
        self.sample_interval_s = max(0.1, sample_interval_s)
        self.intervals: List[Dict[str, float]] = []
        self._active_start: Optional[float] = None
        self._stopped = False

    @property
    def enabled(self) -> bool:
        return self.path is not None

    async def run(self) -> None:
        while not self._stopped:
            self.sample()
            await asyncio.sleep(self.sample_interval_s)

    def stop(self) -> None:
        self._stopped = True

    def close(self, end_time: float) -> None:
        self.sample(now=end_time)
        if self._active_start is not None:
            self.intervals.append({"start": self._active_start, "end": end_time})
            self._active_start = None

    def snapshot(self, now: float) -> List[Dict[str, float]]:
        intervals = list(self.intervals)
        if self._active_start is not None:
            intervals.append({"start": self._active_start, "end": now})
        return intervals

    def sample(self, *, now: Optional[float] = None) -> None:
        if not self.enabled:
            return
        timestamp = time.time() if now is None else now
        state = self._read_state()
        if _is_reconfiguring_state(state):
            if self._active_start is None:
                self._active_start = timestamp
            return
        if self._active_start is not None:
            self.intervals.append({"start": self._active_start, "end": timestamp})
            self._active_start = None

    def _read_state(self) -> Mapping[str, object]:
        assert self.path is not None
        if not self.path.exists():
            return {"state": "serving"}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"state": "serving"}
        if not isinstance(data, Mapping):
            return {"state": "serving"}
        return data


class PromptSampler:
    def __init__(self, args, rng: random.Random):
        self.args = args
        self.rng = rng
        self.examples: List[Tuple[str, List[int]]] = []
        self.separator_token_id = None
        if args.prompt_source == "datasets":
            self._load_dataset_examples(args)

    @property
    def size(self) -> int:
        return len(self.examples)

    def input_ids(self, input_len: int) -> Tuple[List[int], Optional[str]]:
        if self.args.prompt_source == "random-ids":
            token_high = max(self.args.token_low + 1, int(self.args.token_high))
            return (
                [
                    self.rng.randrange(int(self.args.token_low), token_high)
                    for _ in range(input_len)
                ],
                None,
            )
        if not self.examples:
            raise RuntimeError("no dataset examples loaded")
        ids: List[int] = []
        source_labels: List[str] = []
        while len(ids) < input_len:
            label, token_ids = self.rng.choice(self.examples)
            if not token_ids:
                continue
            if ids and self.separator_token_id is not None:
                ids.append(int(self.separator_token_id))
            if len(token_ids) >= input_len:
                max_start = max(0, len(token_ids) - input_len)
                start = self.rng.randrange(max_start + 1) if max_start else 0
                ids.extend(token_ids[start : start + input_len])
            else:
                ids.extend(token_ids)
            source_labels.append(label)
        return ids[:input_len], "+".join(source_labels[:4])

    def _load_dataset_examples(self, args) -> None:
        tokenizer_path = args.tokenizer_path
        if not tokenizer_path:
            raise ValueError("--tokenizer-path is required when --prompt-source=datasets")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "--prompt-source=datasets requires transformers to load the tokenizer"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.separator_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.pad_token_id
        )
        sources = args.dataset_source or DEFAULT_DATASET_SOURCES
        fields = [field.strip() for field in args.dataset_text_fields.split(",") if field.strip()]
        max_examples = max(1, int(args.dataset_samples_per_source))
        for raw_source in sources:
            spec = _parse_dataset_source(raw_source)
            rows = _iter_dataset_rows(
                spec,
                max_examples=max_examples,
                cache_dir=args.dataset_cache_dir,
                trust_remote_code=bool(args.dataset_trust_remote_code),
            )
            loaded = 0
            for row in rows:
                text = _extract_text(row, fields)
                if not text:
                    continue
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    continue
                self.examples.append((spec["label"], [int(token) for token in token_ids]))
                loaded += 1
                if loaded >= max_examples:
                    break
            print(f"loaded dataset source {spec['label']}: {loaded} prompts")
        if not self.examples:
            raise RuntimeError("no prompt text could be loaded from dataset sources")


def _parse_dataset_source(raw: str) -> Dict[str, Optional[str]]:
    if "=" not in raw:
        raise ValueError(f"dataset source must start with LABEL=SOURCE, got {raw!r}")
    label, rest = raw.split("=", 1)
    parts = [part.strip() for part in rest.split(",") if part.strip()]
    if not label.strip() or not parts:
        raise ValueError(f"invalid dataset source: {raw!r}")
    spec: Dict[str, Optional[str]] = {
        "label": label.strip(),
        "source": parts[0],
        "config": None,
        "split": "train",
    }
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"dataset source option expects key=value, got {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in ("config", "split"):
            raise ValueError(f"unknown dataset source option {key!r}")
        spec[key] = value.strip()
    return spec


def _iter_dataset_rows(
    spec: Mapping[str, Optional[str]],
    *,
    max_examples: int,
    cache_dir: Optional[str],
    trust_remote_code: bool,
):
    source = str(spec["source"])
    path = Path(source)
    if path.exists():
        yield from _iter_local_rows(path, max_examples=max_examples)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace dataset sources require the datasets package. "
            "Use local json/jsonl files or install datasets."
        ) from exc
    kwargs = {
        "split": spec.get("split") or "train",
        "streaming": True,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    config = spec.get("config")
    if config:
        dataset = load_dataset(source, config, **kwargs)
    else:
        dataset = load_dataset(source, **kwargs)
    for index, row in enumerate(dataset):
        if index >= max_examples:
            break
        yield row


def _iter_local_rows(path: Path, *, max_examples: int):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= max_examples:
                    break
                if not line.strip():
                    continue
                yield json.loads(line)
        return
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        for key in ("data", "rows", "examples", "train"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"local dataset file must contain a JSON list: {path}")
    for row in data[:max_examples]:
        yield row


def _extract_text(row: object, fields: Sequence[str]) -> str:
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, Mapping):
        return ""
    for field in fields:
        if field not in row:
            continue
        text = _stringify_text_value(row[field])
        if text:
            return text
    return _stringify_text_value(row)


def _stringify_text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        role = value.get("role") or value.get("from")
        content = value.get("content") or value.get("value") or value.get("text")
        if content is not None:
            prefix = f"{role}: " if role else ""
            return f"{prefix}{_stringify_text_value(content)}".strip()
        pieces = [_stringify_text_value(v) for v in value.values()]
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(value, list):
        pieces = [_stringify_text_value(item) for item in value]
        return "\n".join(piece for piece in pieces if piece)
    return str(value).strip()


def _load_trace(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, Mapping):
        raw_windows = raw.get("windows", [])
    else:
        raw_windows = raw
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("trace must be a non-empty JSON list or {'windows': [...]}")
    windows = []
    for index, item in enumerate(raw_windows):
        if not isinstance(item, Mapping):
            raise ValueError(f"trace window {index} is not an object")
        window = dict(item)
        window.setdefault("name", f"window_{index}")
        for key in ("duration_s", "request_rate", "input_len", "output_len"):
            if key not in window:
                raise ValueError(f"trace window {index} missing {key}")
        windows.append(window)
    return windows


async def _run_window(
    *,
    args,
    session: aiohttp.ClientSession,
    url: str,
    window: Mapping[str, object],
    window_index: int,
    benchmark_start: float,
    rng: random.Random,
    prompt_sampler: PromptSampler,
    semaphore: asyncio.Semaphore,
    out,
    total_planned: int,
    overall_progress: Dict[str, int],
) -> List[Dict[str, object]]:
    name = str(window["name"])
    duration_s = float(window["duration_s"])
    request_rate = float(window["request_rate"])
    input_len = int(window["input_len"])
    output_len = int(window["output_len"])
    if duration_s <= 0 or request_rate < 0:
        raise ValueError(f"invalid duration/request_rate for window {name}")
    window_planned = _planned_requests(args, window)
    window_progress = {"sent": 0, "completed": 0, "failed": 0}

    print(
        f"[{window_index}] {name}: duration={duration_s:.1f}s "
        f"rate={request_rate:.3f} req/s input={input_len} output={output_len} "
        f"planned={window_planned}"
    )

    tasks = []
    window_start = time.time()
    progress_task = None
    if args.progress_interval_s > 0:
        progress_task = asyncio.create_task(
            _report_progress(
                window_name=name,
                window_index=window_index,
                window_planned=window_planned,
                total_planned=total_planned,
                window_progress=window_progress,
                overall_progress=overall_progress,
                interval_s=float(args.progress_interval_s),
                heartbeat_s=float(args.progress_heartbeat_s),
                started_at=window_start,
            )
        )
    request_index = 0
    next_arrival = window_start
    try:
        while True:
            now = time.time()
            if now - window_start >= duration_s:
                break
            if args.max_requests_per_window is not None and request_index >= args.max_requests_per_window:
                break
            if request_rate <= 0:
                await asyncio.sleep(min(1.0, max(0.0, duration_s - (now - window_start))))
                continue
            if now < next_arrival:
                await asyncio.sleep(min(next_arrival - now, 0.1))
                continue

            payload, payload_metadata = _make_payload(
                args,
                prompt_sampler,
                input_len,
                output_len,
            )
            task = asyncio.create_task(
                _send_one(
                    session=session,
                    url=url,
                    payload=payload,
                    semaphore=semaphore,
                    benchmark_start=benchmark_start,
                    window_name=name,
                    window_index=window_index,
                    request_index=request_index,
                    request_rate=request_rate,
                    input_len=input_len,
                    output_len=output_len,
                    payload_metadata=payload_metadata,
                    out=out,
                )
            )
            task.add_done_callback(
                lambda done: _mark_completed(done, window_progress, overall_progress)
            )
            tasks.append(task)
            window_progress["sent"] += 1
            overall_progress["sent"] += 1
            request_index += 1
            next_arrival += _arrival_gap(args.arrival_process, request_rate, rng)

        if not tasks:
            return []
        return await asyncio.gather(*tasks)
    finally:
        if progress_task is not None:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        _print_progress_line(
            window_name=name,
            window_index=window_index,
            window_planned=window_planned,
            total_planned=total_planned,
            window_progress=window_progress,
            overall_progress=overall_progress,
            elapsed_s=time.time() - window_start,
            final=True,
        )


def _planned_requests(args, window: Mapping[str, object]) -> int:
    duration_s = float(window["duration_s"])
    request_rate = float(window["request_rate"])
    if request_rate <= 0 or duration_s <= 0:
        return 0
    if args.arrival_process == "poisson":
        planned = int(round(duration_s * request_rate))
    else:
        planned = int(duration_s * request_rate)
        if planned < duration_s * request_rate:
            planned += 1
    if args.max_requests_per_window is not None:
        planned = min(planned, int(args.max_requests_per_window))
    return max(0, planned)


async def _report_progress(
    *,
    window_name: str,
    window_index: int,
    window_planned: int,
    total_planned: int,
    window_progress: Mapping[str, int],
    overall_progress: Mapping[str, int],
    interval_s: float,
    heartbeat_s: float,
    started_at: float,
) -> None:
    last_sent = -1
    last_completed = -1
    last_failed = -1
    last_print_at = started_at
    while True:
        await asyncio.sleep(interval_s)
        now = time.time()
        sent = int(window_progress["sent"])
        completed = int(window_progress["completed"])
        failed = int(window_progress["failed"])
        changed = (
            sent != last_sent
            or completed != last_completed
            or failed != last_failed
        )
        heartbeat = heartbeat_s > 0 and now - last_print_at >= heartbeat_s
        if not changed and not heartbeat:
            continue
        _print_progress_line(
            window_name=window_name,
            window_index=window_index,
            window_planned=window_planned,
            total_planned=total_planned,
            window_progress=window_progress,
            overall_progress=overall_progress,
            elapsed_s=now - started_at,
            final=False,
        )
        last_sent = sent
        last_completed = completed
        last_failed = failed
        last_print_at = now


def _print_progress_line(
    *,
    window_name: str,
    window_index: int,
    window_planned: int,
    total_planned: int,
    window_progress: Mapping[str, int],
    overall_progress: Mapping[str, int],
    elapsed_s: float,
    final: bool,
) -> None:
    prefix = "  final progress" if final else "  progress"
    print(
        f"{prefix} [{window_index}] {window_name}: "
        f"sent={window_progress['sent']}/{window_planned} "
        f"done={window_progress['completed']}/{window_progress['sent']} "
        f"failed={window_progress['failed']} "
        f"overall_sent={overall_progress['sent']}/{total_planned} "
        f"overall_done={overall_progress['completed']}/{overall_progress['sent']} "
        f"elapsed={elapsed_s:.1f}s",
        flush=True,
    )


def _mark_completed(
    task: asyncio.Task,
    window_progress: Dict[str, int],
    overall_progress: Dict[str, int],
) -> None:
    window_progress["completed"] += 1
    overall_progress["completed"] += 1
    failed = False
    if task.cancelled():
        failed = True
    else:
        try:
            result = task.result()
            failed = not bool(result.get("ok"))
        except Exception:
            failed = True
    if failed:
        window_progress["failed"] += 1
        overall_progress["failed"] += 1


def _make_payload(
    args,
    prompt_sampler: PromptSampler,
    input_len: int,
    output_len: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    input_ids, dataset_source = prompt_sampler.input_ids(input_len)
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": bool(args.ignore_eos),
        },
    }
    metadata = {
        "prompt_source": args.prompt_source,
        "dataset_source": dataset_source,
    }
    return payload, metadata


def _arrival_gap(process: str, request_rate: float, rng: random.Random) -> float:
    if request_rate <= 0:
        return float("inf")
    if process == "poisson":
        return rng.expovariate(request_rate)
    return 1.0 / request_rate


async def _send_one(
    *,
    session: aiohttp.ClientSession,
    url: str,
    payload: Mapping[str, object],
    semaphore: asyncio.Semaphore,
    benchmark_start: float,
    window_name: str,
    window_index: int,
    request_index: int,
    request_rate: float,
    input_len: int,
    output_len: int,
    payload_metadata: Mapping[str, object],
    out,
) -> Dict[str, object]:
    async with semaphore:
        start = time.time()
        result: Dict[str, object] = {
            "window": window_name,
            "window_index": window_index,
            "request_index": request_index,
            "target_request_rate": request_rate,
            "input_len": input_len,
            "output_len": output_len,
            **payload_metadata,
            "start_time": start,
            "start_s": start - benchmark_start,
        }
        try:
            async with session.post(url, json=payload) as response:
                body = await response.text()
                end = time.time()
                result.update(
                    {
                        "end_time": end,
                        "end_s": end - benchmark_start,
                        "latency_s": end - start,
                        "status": response.status,
                        "ok": 200 <= response.status < 300,
                        "response_bytes": len(body.encode("utf-8")),
                    }
                )
                if not result["ok"]:
                    result["error"] = body[:1000]
        except Exception as exc:
            end = time.time()
            result.update(
                {
                    "end_time": end,
                    "end_s": end - benchmark_start,
                    "latency_s": end - start,
                    "status": None,
                    "ok": False,
                    "error": repr(exc),
                }
            )
        out.write(json.dumps(result, sort_keys=True) + "\n")
        out.flush()
        return result


def _summarize(
    windows: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
    *,
    total_elapsed_s: float,
    benchmark_start: float,
    drain_intervals: Sequence[Mapping[str, float]],
) -> Dict[str, object]:
    by_window: Dict[str, List[Mapping[str, object]]] = {}
    for result in results:
        by_window.setdefault(str(result["window"]), []).append(result)
    successful_results = [r for r in results if r.get("ok")]
    output_tokens = sum(int(r.get("output_len", 0)) for r in successful_results)
    return {
        "total_requests": len(results),
        "successful_requests": len(successful_results),
        "failed_requests": sum(1 for r in results if not r.get("ok")),
        "total_elapsed_s": total_elapsed_s,
        "goodput_req_s": _safe_div(len(successful_results), total_elapsed_s),
        "output_tok_s": _safe_div(output_tokens, total_elapsed_s),
        "drain_intervals": [
            {
                "start_s": float(interval["start"]) - benchmark_start,
                "end_s": float(interval["end"]) - benchmark_start,
                "duration_s": max(0.0, float(interval["end"]) - float(interval["start"])),
            }
            for interval in drain_intervals
        ],
        "windows": [
            _window_summary(
                str(window["name"]),
                by_window.get(str(window["name"]), []),
                drain_intervals=drain_intervals,
            )
            for window in windows
        ],
    }


def _window_summary(
    name: str,
    results: Sequence[Mapping[str, object]],
    *,
    drain_intervals: Sequence[Mapping[str, float]] = (),
) -> Dict[str, object]:
    latencies = [float(r["latency_s"]) for r in results if r.get("ok") and "latency_s" in r]
    active_latencies = [
        _active_latency(r, drain_intervals)
        for r in results
        if r.get("ok")
        and "latency_s" in r
        and "start_time" in r
        and "end_time" in r
    ]
    failed = sum(1 for r in results if not r.get("ok"))
    successful_results = [r for r in results if r.get("ok")]
    output_tokens = sum(int(r.get("output_len", 0)) for r in successful_results)
    start_times = [float(r["start_time"]) for r in results if "start_time" in r]
    end_times = [float(r["end_time"]) for r in results if "end_time" in r]
    elapsed_s = max(end_times) - min(start_times) if start_times and end_times else 0.0
    excluded_drain_s = (
        _overlap_duration(min(start_times), max(end_times), drain_intervals)
        if start_times and end_times
        else 0.0
    )
    active_elapsed_s = max(0.0, elapsed_s - excluded_drain_s)
    summary: Dict[str, object] = {
        "name": name,
        "requests": len(results),
        "successful_requests": len(latencies),
        "failed_requests": failed,
        "elapsed_s": elapsed_s,
        "excluded_drain_s": excluded_drain_s,
        "active_elapsed_s": active_elapsed_s,
        "goodput_req_s": _safe_div(len(successful_results), elapsed_s),
        "output_tok_s": _safe_div(output_tokens, elapsed_s),
        "active_goodput_req_s": _safe_div(len(successful_results), active_elapsed_s),
        "active_output_tok_s": _safe_div(output_tokens, active_elapsed_s),
    }
    if latencies:
        summary.update(
            {
                "latency_avg_s": statistics.fmean(latencies),
                "latency_p50_s": _percentile(latencies, 50),
                "latency_p95_s": _percentile(latencies, 95),
                "latency_p99_s": _percentile(latencies, 99),
            }
        )
    if active_latencies:
        summary.update(
            {
                "active_latency_avg_s": statistics.fmean(active_latencies),
                "active_latency_p50_s": _percentile(active_latencies, 50),
                "active_latency_p95_s": _percentile(active_latencies, 95),
                "active_latency_p99_s": _percentile(active_latencies, 99),
            }
        )
    return summary


def _print_window_summary(
    window: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    *,
    drain_intervals: Sequence[Mapping[str, float]] = (),
) -> None:
    summary = _window_summary(
        str(window["name"]),
        results,
        drain_intervals=drain_intervals,
    )
    active_elapsed_s = float(summary.get("active_elapsed_s", summary.get("elapsed_s", 0.0)))
    excluded_drain_s = float(summary.get("excluded_drain_s", 0.0))
    print(
        f"  done {summary['name']}: requests={summary['requests']} "
        f"ok={summary['successful_requests']} failed={summary['failed_requests']} "
        f"elapsed={float(summary.get('elapsed_s', 0.0)):.3f}s "
        f"active_elapsed={active_elapsed_s:.3f}s "
        f"excluded_drain={excluded_drain_s:.3f}s "
        f"active_goodput={float(summary.get('active_goodput_req_s', 0.0)):.3f}req/s "
        f"active_out_tok={float(summary.get('active_output_tok_s', 0.0)):.1f}tok/s "
        f"p95={float(summary.get('latency_p95_s', 0.0)):.3f}s "
        f"active_p95={float(summary.get('active_latency_p95_s', 0.0)):.3f}s"
    )


def _print_overall_summary(summary: Mapping[str, object]) -> None:
    print("-" * 100)
    print(
        f"total={summary['total_requests']} ok={summary['successful_requests']} "
        f"failed={summary['failed_requests']} "
        f"elapsed={float(summary.get('total_elapsed_s', 0.0)):.3f}s "
        f"goodput={float(summary.get('goodput_req_s', 0.0)):.3f}req/s "
        f"out_tok={float(summary.get('output_tok_s', 0.0)):.1f}tok/s"
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _overlap_duration(
    start: float,
    end: float,
    intervals: Sequence[Mapping[str, float]],
) -> float:
    total = 0.0
    for interval in intervals:
        interval_start = float(interval["start"])
        interval_end = float(interval["end"])
        total += max(0.0, min(end, interval_end) - max(start, interval_start))
    return min(max(0.0, end - start), total)


def _active_latency(
    result: Mapping[str, object],
    intervals: Sequence[Mapping[str, float]],
) -> float:
    start = float(result["start_time"])
    end = float(result["end_time"])
    latency = float(result["latency_s"])
    excluded = _overlap_duration(start, end, intervals)
    return max(0.0, latency - excluded)


def _is_reconfiguring_state(state: Mapping[str, object]) -> bool:
    phase = state.get("phase")
    if phase is not None:
        return str(phase) == "reconfiguring"
    return str(state.get("state", "serving")) == "draining"


def _percentile(values: Iterable[float], percentile: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
