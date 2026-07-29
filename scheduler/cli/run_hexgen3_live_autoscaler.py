#!/usr/bin/env python3
"""HexGen-3 live autoscaling planner from runtime metrics.

This is the small online bridge for the release demo:

  metrics json/jsonl -> WorkloadProfile -> scheduler -> RuntimeLaunchSpec

It writes the generated plan, optionally applies runtime windows, and keeps the
mini-lb stable across scale reconfigurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _autoscaling_stability import (  # noqa: E402
    StabilityValidationConfig,
    stabilize_autoscaling_plan,
)
from _runtime_apply import (  # noqa: E402
    ApplyRuntimeOptions,
    apply_runtime_window,
)
from simulator.scheduling import (  # noqa: E402
    AutoscalingConfig,
    GlobalSchedulerConfig,
    HexGenSchedulingFramework,
    LocalSchedulerConfig,
    WORKER_TYPES,
    WorkloadProfile,
    afd_deployment_spec_to_runtime_launch_spec,
    plan_to_afd_deployment_spec,
    plan_to_dict,
)
from simulator.scheduling.types import AllocationMatrix, DeploymentPlan  # noqa: E402


INITIAL_METRICS = {
    "source": "initial_default",
    "arrival_rate_rps": 1.0,
    "avg_input_tokens": 2048,
    "avg_output_tokens": 512,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dry-run HexGen-3 autoscaling planner driven by runtime metrics"
    )
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory containing runtime metrics json/jsonl files",
    )
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--min-arrival-rate", type=float, default=1e-6)
    parser.add_argument(
        "--workload-shape-max-age-s",
        type=float,
        default=300.0,
        help="Maximum age of cached attention input/output averages.",
    )

    parser.add_argument("--capacity", default='{"NVDA:H100:SXM": 3}')
    parser.add_argument(
        "--initial-allocation",
        default='{"pre":1,"attn":1,"ffn":1}',
        help=(
            "Initial runtime GPU allocation by worker. Values are placed on the "
            "first hardware type with enough capacity."
        ),
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/model",
        help="Model path used in generated sglang.launch_server commands",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--stability-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--model-size-billions", type=float, default=30.0)
    parser.add_argument("--enable-ep", action="store_true")
    parser.add_argument("--num-experts", type=int, default=1)
    parser.add_argument("--cost-aware", action="store_true")
    parser.add_argument("--stability-window-s", type=float, default=300.0)
    parser.add_argument("--reload-bandwidth-gbps", type=float, default=600.0)
    parser.add_argument("--model-size-gb", type=float, default=60.0)
    parser.add_argument("--target-utilization", type=float, default=0.75)
    parser.add_argument("--hysteresis", type=float, default=0.08)
    parser.add_argument("--decode-worker-gpu-choices", default="1,2,4,8")
    parser.add_argument("--stability-search-max-rounds", type=int, default=3)
    parser.add_argument("--stability-search-max-candidates", type=int, default=16)
    parser.add_argument("--stability-search-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--scale-in-backlog-threshold",
        type=float,
        default=None,
        help=(
            "Maximum queued requests that still allow scale-in. By default, "
            "one polling interval of arrivals is allowed."
        ),
    )
    parser.add_argument(
        "--kv-transfer-bandwidth-gbps",
        type=float,
        default=100.0,
        help="Estimator KV transfer bandwidth in GB/s.",
    )
    parser.add_argument(
        "--activation-bandwidth-gbps",
        type=float,
        default=100.0,
        help="Estimator activation transfer bandwidth in GB/s between AFD stages.",
    )

    parser.add_argument("--afd-micro-batch", type=int, default=2)
    parser.add_argument("--afd-sched-host", default="127.0.0.1")
    parser.add_argument("--dmlc-ps-root-uri", default=None)
    parser.add_argument("--dmlc-node-host", default=None)
    parser.add_argument("--mlc-interface", default="bond2")
    parser.add_argument("--disaggregation-ib-device", default="mlx5_bond_1")
    parser.add_argument("--runtime-host", default="127.0.0.1")
    parser.add_argument("--runtime-base-port", type=int, default=30001)
    parser.add_argument("--lb-host", default="127.0.0.1")
    parser.add_argument("--lb-port", type=int, default=30000)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--output", default="/tmp/hexgen3_live_autoscaling_plan.json")
    parser.add_argument(
        "--apply-runtime",
        action="store_true",
        help="Launch/restart runtime only when the decision is initial or scale_changed.",
    )
    parser.add_argument(
        "--runtime-cwd",
        default=None,
        help="Override cwd for launched runtime processes.",
    )
    parser.add_argument(
        "--runtime-log-dir",
        default="/tmp/hexgen3_runtime_logs",
        help="Directory for runtime stdout/stderr logs when --apply-runtime is used.",
    )
    parser.add_argument(
        "--runtime-pid-file",
        default="/tmp/hexgen3_runtime_pids.json",
        help="Pid file used by runtime apply/stop.",
    )
    parser.add_argument(
        "--cleanup-stale-runtime",
        action="store_true",
        default=_env_flag("HEXGEN3_CLEANUP_STALE_RUNTIME"),
        help=(
            "Before launching, scan and kill stale local sglang/compile-worker "
            "processes. Disabled by default; useful after a crashed run."
        ),
    )
    parser.add_argument("--drain-state-file", default="/tmp/afd_runtime_state.json")
    parser.add_argument("--drain-wait-timeout-s", type=float, default=600.0)
    parser.add_argument("--drain-timeout-s", type=float, default=300.0)
    parser.add_argument("--drain-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--runtime-startup-wait-s", type=float, default=2.0)
    parser.add_argument("--runtime-ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--runtime-ready-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--runtime-ready-log-interval-s", type=float, default=15.0)
    parser.add_argument(
        "--apply-cooldown-s",
        type=float,
        default=20.0,
        help="Seconds to wait after runtime apply before allowing another scale decision.",
    )
    parser.add_argument(
        "--scale-confirmations",
        type=int,
        default=2,
        help=(
            "Require a scale decision this many consecutive steps. Scale-out "
            "targets may vary while the expansion signal remains active; "
            "scale-in and same-size reconfiguration targets must match exactly."
        ),
    )
    parser.add_argument(
        "--max-scale-applies",
        default=None,
        help=(
            "Maximum number of scale_changed runtime applies. Use 0/never/off "
            "to disable reconfiguration after initial launch; use -1/unlimited/"
            "forever/inf or omit this option for unlimited reconfiguration."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capacity = _parse_capacity(args.capacity)
    initial_allocation = _parse_initial_allocation(args.initial_allocation, capacity)
    framework = _build_framework(args)
    _write_runtime_state(
        args.drain_state_file,
        "draining" if args.apply_runtime else "serving",
    )

    previous_plan: Optional[DeploymentPlan] = None
    windows: List[Dict[str, object]] = []
    last_metrics_key: Optional[str] = None
    cooldown_until_s = 0.0
    pending_scale_signature: Optional[str] = None
    pending_scale_count = 0
    scale_applies = 0
    max_scale_applies = _max_scale_applies(args)
    workload_shape_state: Dict[str, Tuple[float, float]] = {}
    step = 0
    while True:
        if previous_plan is None:
            metrics = _initial_runtime_metrics()
            print(
                f"[{step}] initial: launching runtime from initial_allocation="
                f"{initial_allocation.values}"
            )
        else:
            try:
                metrics, last_metrics_key, metrics_count = _read_unread_metrics(
                    args,
                    last_metrics_key,
                )
            except FileNotFoundError as exc:
                print(f"[{step}] hold: {exc}; waiting for metrics")
                step += 1
                _sleep(args)
                continue
            if metrics is None:
                print(f"[{step}] hold: no unread metrics samples")
                step += 1
                _sleep(args)
                continue
            if metrics_count > 1:
                print(f"[{step}] aggregated {metrics_count} unread metrics samples")
        try:
            workload = _workload_from_metrics(
                args,
                metrics,
                shape_state=(workload_shape_state if previous_plan is not None else None),
            )
        except ValueError as exc:
            print(f"[{step}] hold: {exc}")
            step += 1
            _sleep(args)
            continue
        if workload.arrival_rate < args.min_arrival_rate:
            print(
                f"[{step}] hold: arrival_rate_rps={workload.arrival_rate:.6f} "
                f"below --min-arrival-rate"
            )
            step += 1
            _sleep(args)
            continue
        if previous_plan is not None and time.time() < cooldown_until_s:
            remaining_s = cooldown_until_s - time.time()
            print(f"[{step}] hold: apply cooldown active ({remaining_s:.1f}s remaining)")
            previous_plan = _evaluate_existing_plan_for_workload(
                framework,
                workload,
                previous_plan,
            )
            step += 1
            _sleep(args)
            continue

        window, planned_next_plan = _plan_one_step(
            args=args,
            framework=framework,
            workload=workload,
            metrics=metrics,
            index=step,
            previous_plan=previous_plan,
            capacity=capacity,
            initial_allocation=initial_allocation,
        )
        windows.append(window)
        result = _result_payload(
            args,
            windows,
            capacity,
            initial_allocation,
        )
        _write_json(args.output, result)
        decision = str(window["reschedule_decision"])
        _print_step_summary(window)
        _print_stability_search_summary(window)
        if decision != "scale_changed":
            pending_scale_signature = None
            pending_scale_count = 0

        if _should_apply_runtime(args, window):
            if (
                decision == "scale_changed"
                and max_scale_applies is not None
                and scale_applies >= max_scale_applies
            ):
                print(
                    "hold: max scale applies reached "
                    f"({scale_applies}/{max_scale_applies}); skip runtime apply"
                )
                window["apply_status"] = "skipped_max_scale_applies"
                _write_json(args.output, result)
                previous_plan = _evaluate_existing_plan_for_workload(
                    framework,
                    workload,
                    previous_plan,
                )
                step += 1
                _sleep(args)
                continue
            if decision == "scale_changed" and args.scale_confirmations > 1:
                scale_direction = _scale_plan_direction(
                    previous_plan,
                    planned_next_plan,
                )
                signature = _scale_confirmation_signature(
                    previous_plan,
                    planned_next_plan,
                )
                if signature == pending_scale_signature:
                    pending_scale_count += 1
                else:
                    pending_scale_signature = signature
                    pending_scale_count = 1
                required = max(1, int(args.scale_confirmations))
                if pending_scale_count < required:
                    confirmation_target = (
                        "scale-out remains necessary"
                        if scale_direction == "scale_out"
                        else "the same target deployment is selected again"
                    )
                    print(
                        "pending scale confirmation: "
                        f"{pending_scale_count}/{required}; "
                        f"hold current runtime until {confirmation_target}"
                    )
                    window["apply_status"] = "pending_scale_confirmation"
                    window["scale_confirmation"] = {
                        "count": pending_scale_count,
                        "required": required,
                        "direction": scale_direction,
                    }
                    _write_json(args.output, result)
                    previous_plan = _evaluate_existing_plan_for_workload(
                        framework,
                        workload,
                        previous_plan,
                    )
                    step += 1
                    _sleep(args)
                    continue
            if args.apply_runtime:
                if decision == "initial":
                    _write_runtime_state(
                        args.drain_state_file,
                        "draining",
                        phase="initial_apply",
                    )
                elif decision == "scale_changed":
                    drained = _drain_before_apply(args)
                    if not drained:
                        print(
                            "skip apply: runtime did not drain before timeout; "
                            "keeping pending plan for the next polling step"
                        )
                        window["apply_status"] = "skipped_drain_timeout"
                        _write_json(args.output, result)
                        _write_runtime_state(args.drain_state_file, "serving")
                        previous_plan = _evaluate_existing_plan_for_workload(
                            framework,
                            workload,
                            previous_plan,
                        )
                        step += 1
                        _sleep(args)
                        continue
                    _write_runtime_state(
                        args.drain_state_file,
                        "draining",
                        phase="reconfiguring",
                    )
            status = _apply_runtime_for_window(args, window, apply=args.apply_runtime)
            if status != 0:
                return status
            if args.apply_runtime:
                if not _wait_runtime_ready(args, window):
                    print(
                        "runtime readiness check timed out; keep drain state "
                        "as draining to avoid forwarding requests to incomplete runtime"
                    )
                    window["apply_status"] = "runtime_ready_timeout"
                    _write_json(args.output, result)
                    return 1
                cooldown_until_s = time.time() + _apply_cooldown_s(args)
                _write_runtime_state(args.drain_state_file, "serving")
                window["apply_status"] = "applied"
            _write_json(args.output, result)
            if decision == "scale_changed":
                scale_applies += 1
                pending_scale_signature = None
                pending_scale_count = 0
        elif args.apply_runtime and decision != "hold_allocation_unchanged":
            print(f"runtime apply: skipped decision={decision}")

        previous_plan = planned_next_plan
        step += 1
        _sleep(args)

    return 0


def _should_apply_runtime(args, window: Mapping[str, object]) -> bool:
    return bool(args.apply_runtime) and str(window["reschedule_decision"]) in {
        "initial",
        "scale_changed",
    }


def _apply_runtime_for_window(args, window: Mapping[str, object], *, apply: bool) -> int:
    if not apply:
        return 0
    return apply_runtime_window(
        ApplyRuntimeOptions(
            plan_path=args.output,
            window=window,
            gpu_ids=args.gpu_ids,
            cwd=args.runtime_cwd,
            log_dir=args.runtime_log_dir,
            pid_file=args.runtime_pid_file,
            extra_env=_runtime_extra_env(args),
            lb_host=args.lb_host,
            lb_port=args.lb_port,
            drain_state_file=args.drain_state_file,
            drain_wait_timeout_s=args.drain_wait_timeout_s,
            startup_wait_s=args.runtime_startup_wait_s,
            restart_lb=str(window["reschedule_decision"]) == "initial",
            cleanup_stale_runtime=bool(args.cleanup_stale_runtime),
        )
    )


def _runtime_extra_env(args) -> Dict[str, str]:
    extra_env: Dict[str, str] = {}
    if args.metrics_dir:
        extra_env["SGLANG_ENABLE_AFD_METRICS"] = "1"
        extra_env["SGLANG_AFD_METRICS_DIR"] = args.metrics_dir
    return extra_env


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _evaluate_existing_plan_for_workload(
    framework: HexGenSchedulingFramework,
    workload: WorkloadProfile,
    previous_plan: DeploymentPlan,
) -> DeploymentPlan:
    return framework.evaluate_allocation(
        workload,
        previous_plan.allocation,
        previous_parallelism=previous_plan.parallelism,
    )


def _drain_before_apply(args) -> bool:
    _write_runtime_state(args.drain_state_file, "draining", phase="drain_wait")
    deadline = time.time() + max(0.0, float(args.drain_timeout_s))
    poll_interval_s = max(0.1, float(args.drain_poll_interval_s))
    log_interval_s = max(poll_interval_s, 15.0)
    next_log_at = time.time()
    last_running = 0.0
    last_waiting = 0.0
    print(
        f"draining runtime before apply: state_file={args.drain_state_file} "
        f"timeout={args.drain_timeout_s:.1f}s"
    )
    while time.time() <= deadline:
        try:
            metrics = _read_metrics_or_default(args)
        except FileNotFoundError as exc:
            if time.time() >= next_log_at:
                print(f"  drain wait: {exc}")
                next_log_at = time.time() + log_interval_s
            time.sleep(poll_interval_s)
            continue
        running = _first_float(
            metrics,
            ("running_requests", "num_running_requests", "running_reqs"),
            default=0.0,
        )
        waiting = _first_float(
            metrics,
            ("waiting_requests", "num_waiting_requests", "waiting_reqs"),
            default=0.0,
        )
        last_running = running
        last_waiting = waiting
        if time.time() >= next_log_at:
            print(f"  drain wait: running={running:.0f} waiting={waiting:.0f}")
            next_log_at = time.time() + log_interval_s
        if running <= 0 and waiting <= 0:
            print("runtime drained; applying pending runtime plan")
            return True
        time.sleep(poll_interval_s)
    print(
        "drain timed out: "
        f"last_running={last_running:.0f} last_waiting={last_waiting:.0f}"
    )
    return False


def _wait_runtime_ready(args, window: Mapping[str, object]) -> bool:
    probes = _runtime_health_probes(window)
    if not probes:
        print("runtime readiness: no HTTP endpoints found in runtime_launch")
        return True
    deadline = time.time() + max(0.0, float(args.runtime_ready_timeout_s))
    poll_interval_s = max(0.1, float(args.runtime_ready_poll_interval_s))
    log_interval_s = max(poll_interval_s, float(args.runtime_ready_log_interval_s))
    next_log_at = time.time()
    pending = dict(probes)
    print(
        "waiting for runtime readiness: "
        f"endpoints={len(pending)} timeout={args.runtime_ready_timeout_s:.1f}s"
    )
    while pending and time.time() <= deadline:
        for name, url in list(pending.items()):
            if _probe_http_ready(url):
                pending.pop(name, None)
        if not pending:
            print("runtime readiness: all endpoints ready")
            return True
        now = time.time()
        if now >= next_log_at:
            print(f"  readiness wait: pending={', '.join(sorted(pending))}")
            next_log_at = now + log_interval_s
        time.sleep(poll_interval_s)
    for name, url in sorted(pending.items()):
        print(f"  not ready {name}: {url}")
    return False


def _runtime_health_probes(window: Mapping[str, object]) -> Dict[str, str]:
    launch = window.get("runtime_launch", {})
    if not isinstance(launch, Mapping):
        return {}
    probes: Dict[str, str] = {}
    for process in launch.get("processes", []):
        if not isinstance(process, Mapping):
            continue
        role = str(process.get("role", ""))
        if role not in {"prefill", "attention", "ffn"}:
            continue
        command = [str(part) for part in process.get("command", [])]
        host = _command_option(command, "--host", default="127.0.0.1")
        port = _command_option(command, "--port", default=None)
        if not port:
            continue
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        probes[str(process.get("name", role))] = (
            f"http://{host}:{port}/health_ready"
        )
    return probes


def _command_option(
    command: Sequence[str],
    option: str,
    *,
    default: Optional[str],
) -> Optional[str]:
    try:
        index = command.index(option)
    except ValueError:
        return default
    value_index = index + 1
    if value_index >= len(command):
        return default
    return command[value_index]


def _probe_http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            return 200 <= int(response.status) < 400
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _apply_cooldown_s(args) -> float:
    return max(0.0, float(args.apply_cooldown_s))


def _max_scale_applies(args) -> Optional[int]:
    raw = args.max_scale_applies
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "unlimited", "forever", "inf", "infinite", "none"}:
        return None
    if value in {"never", "off", "disable", "disabled"}:
        return 0
    count = int(value)
    return None if count < 0 else count


def _scale_plan_signature(plan: DeploymentPlan) -> str:
    parallelism = []
    for (worker, hardware), strategy in sorted(plan.parallelism.items()):
        parallelism.append(
            {
                "worker": worker,
                "hardware": hardware,
                "replicas": strategy.as_tuple(),
            }
        )
    payload = {
        "allocation": plan.allocation.as_key(),
        "parallelism": parallelism,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _scale_plan_direction(
    current_plan: DeploymentPlan,
    target_plan: DeploymentPlan,
) -> str:
    current_gpus = current_plan.allocation.total_gpus()
    target_gpus = target_plan.allocation.total_gpus()
    if target_gpus > current_gpus:
        return "scale_out"
    if target_gpus < current_gpus:
        return "scale_in"
    return "reconfigure"


def _scale_confirmation_signature(
    current_plan: DeploymentPlan,
    target_plan: DeploymentPlan,
) -> str:
    direction = _scale_plan_direction(current_plan, target_plan)
    if direction == "scale_out":
        return direction
    return f"{direction}:{_scale_plan_signature(target_plan)}"


def _write_runtime_state(
    path: Optional[str],
    state: str,
    *,
    phase: Optional[str] = None,
) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "timestamp": time.time()}
    if phase:
        payload["phase"] = phase
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, output)
    phase_text = f" phase={phase}" if phase else ""
    print(f"runtime state: {state}{phase_text} ({output})")


def _plan_one_step(
    *,
    args,
    framework: HexGenSchedulingFramework,
    workload: WorkloadProfile,
    metrics: Mapping[str, object],
    index: int,
    previous_plan: Optional[DeploymentPlan],
    capacity: Mapping[str, int],
    initial_allocation: AllocationMatrix,
) -> Tuple[Dict[str, object], DeploymentPlan]:
    started_at = time.perf_counter()
    allow_scale_in = True
    waiting_requests = 0.0
    backlog_threshold = None
    if previous_plan is None:
        plan = framework.evaluate_allocation(workload, initial_allocation)
        plan.metadata.setdefault("autoscaling", {})
        plan.metadata["initial_runtime_allocation"] = initial_allocation.values
        decision = "initial"
        worker_expansion = {worker: 1.0 for worker in WORKER_TYPES}
    else:
        allow_scale_in, waiting_requests, backlog_threshold = _scale_in_guard(
            args,
            workload,
            metrics,
        )
        plan, decision, worker_expansion = _plan_existing_allocation(
            framework,
            workload,
            previous_plan,
            capacity,
            allow_scale_in=allow_scale_in,
            validation_config=StabilityValidationConfig(
                max_rounds=args.stability_search_max_rounds,
                max_candidates=args.stability_search_max_candidates,
                timeout_s=args.stability_search_timeout_s,
            ),
        )
    elapsed_s = time.perf_counter() - started_at

    afd_spec = plan_to_afd_deployment_spec(
        plan,
        model_path=args.model_path,
        afd_micro_batch=args.afd_micro_batch,
        disaggregation_ib_device=args.disaggregation_ib_device,
        host=args.runtime_host,
        base_port=args.runtime_base_port,
        afd_sched_host=args.afd_sched_host,
        dmlc_ps_root_uri=args.dmlc_ps_root_uri,
        dmlc_node_host=args.dmlc_node_host,
        mlc_interface=args.mlc_interface,
    )
    launch_spec = afd_deployment_spec_to_runtime_launch_spec(afd_spec)
    return (
        {
            "index": index,
            "name": f"step_{index}",
            "timestamp": time.time(),
            "elapsed_s": elapsed_s,
            "metrics": dict(metrics),
            "workload": {
                "arrival_rate": workload.arrival_rate,
                "mean_input": workload.mean_input,
                "mean_output": workload.mean_output,
                "mean_decode_context": workload.mean_decode_context,
                "max_batch_size": workload.max_batch_size,
            },
            "capacity": dict(capacity),
            "worker_expansion": dict(worker_expansion),
            "reschedule_decision": decision,
            "scale_in_guard": {
                "allowed": allow_scale_in,
                "waiting_requests": waiting_requests,
                "backlog_threshold": backlog_threshold,
            },
            "scheduling_plan": plan_to_dict(plan),
            "afd_deployment": afd_spec.as_dict(),
            "runtime_launch": launch_spec.as_dict(),
        },
        plan,
    )


def _plan_existing_allocation(
    framework: HexGenSchedulingFramework,
    workload: WorkloadProfile,
    previous_plan: DeploymentPlan,
    capacity: Mapping[str, int],
    *,
    allow_scale_in: bool = True,
    validation_config: Optional[StabilityValidationConfig] = None,
) -> Tuple[DeploymentPlan, str, Dict[str, float]]:
    current_plan = _evaluate_existing_plan_for_workload(
        framework,
        workload,
        previous_plan,
    )
    scaled = framework.proportional_scale_allocation(workload, current_plan, capacity)
    worker_expansion = framework.worker_expansion_factors(workload, current_plan)
    if scaled.as_key() != current_plan.allocation.as_key():
        proposed_plan = framework.reschedule(workload, current_plan, capacity)
        if proposed_plan.allocation.as_key() == current_plan.allocation.as_key():
            return proposed_plan, "hold_allocation_unchanged", worker_expansion
        plan, validation = stabilize_autoscaling_plan(
            framework,
            workload,
            current_plan,
            proposed_plan,
            capacity,
            config=validation_config,
            allow_scale_in=allow_scale_in,
        )
        proposal_autoscaling = proposed_plan.metadata.get("autoscaling", {})
        plan.metadata["autoscaling"] = (
            dict(proposal_autoscaling)
            if isinstance(proposal_autoscaling, Mapping)
            else {}
        )
        plan.metadata["autoscaling"]["stability_validation"] = validation
        decision = (
            "scale_changed"
            if plan.allocation.as_key() != current_plan.allocation.as_key()
            else "hold_allocation_unchanged"
        )
        return plan, decision, worker_expansion
    return current_plan, "hold_allocation_unchanged", worker_expansion


def _scale_in_guard(
    args,
    workload: WorkloadProfile,
    metrics: Mapping[str, object],
) -> Tuple[bool, float, float]:
    waiting_requests = max(
        0.0,
        _first_float(
            metrics,
            ("waiting_requests", "num_waiting_requests", "waiting_reqs"),
            default=0.0,
        ),
    )
    configured_threshold = getattr(args, "scale_in_backlog_threshold", None)
    if configured_threshold is None:
        polling_window_s = max(1.0, float(getattr(args, "poll_interval_s", 30.0)))
        backlog_threshold = max(1.0, workload.arrival_rate * polling_window_s)
    else:
        backlog_threshold = max(0.0, float(configured_threshold))
    return waiting_requests <= backlog_threshold, waiting_requests, backlog_threshold


def _build_framework(args) -> HexGenSchedulingFramework:
    return HexGenSchedulingFramework(
        model_id=args.model_path,
        local_config=LocalSchedulerConfig(
            enumerate_non_uniform=True,
            enable_expert_parallel=args.enable_ep,
            num_experts=args.num_experts,
            cost_aware=args.cost_aware,
            stability_window_s=args.stability_window_s,
            reload_bandwidth_gbps=args.reload_bandwidth_gbps,
            model_size_gb=args.model_size_gb,
        ),
        global_config=GlobalSchedulerConfig(
            iterations=args.iterations,
            stability_iterations=args.stability_iterations,
            seed=args.seed,
            block_size=args.block_size,
            model_size_billions=args.model_size_billions,
        ),
        autoscaling_config=AutoscalingConfig(
            target_utilization=args.target_utilization,
            hysteresis=args.hysteresis,
            min_scale_factor=0.0,
            max_scale_factor=float("inf"),
            decode_worker_gpu_choices=tuple(_parse_int_list(args.decode_worker_gpu_choices)),
            global_search_after_scaling=False,
        ),
        kv_transfer_bandwidth_gbps=args.kv_transfer_bandwidth_gbps,
        activation_bandwidth_gbps=args.activation_bandwidth_gbps,
    )


def _read_metrics_or_default(args) -> Dict[str, object]:
    if not args.metrics_dir:
        return _initial_runtime_metrics()
    paths = _select_metrics_files(args.metrics_dir)
    if len(paths) > 1 and all(path.suffix == ".jsonl" for path in paths):
        samples = [_read_latest_json(path) for path in paths]
        return _aggregate_parallel_metrics_samples(samples, len(samples))
    return _read_latest_json(paths[-1])


def _initial_runtime_metrics() -> Dict[str, object]:
    metrics = dict(INITIAL_METRICS)
    metrics["source"] = "initial_runtime"
    metrics["timestamp"] = time.time()
    return metrics


def _read_unread_metrics(
    args,
    last_metrics_key: Optional[str],
) -> Tuple[Optional[Dict[str, object]], Optional[str], int]:
    if not args.metrics_dir:
        metrics = _initial_runtime_metrics()
        return metrics, _metrics_sample_key(metrics), 1

    paths = _select_metrics_files(args.metrics_dir)
    if len(paths) > 1 and all(path.suffix == ".jsonl" for path in paths):
        return _read_unread_metrics_from_files(paths, last_metrics_key)

    path = paths[-1]
    if path.suffix != ".jsonl":
        metrics = _read_latest_json(path)
        metrics_key = _metrics_sample_key(metrics)
        if metrics_key == last_metrics_key:
            return None, last_metrics_key, 0
        return metrics, metrics_key, 1

    samples = _read_jsonl_samples(path)
    if not samples:
        raise ValueError(f"no valid JSON lines found in {path}")
    if last_metrics_key is None:
        latest = samples[-1]
        return latest, _metrics_sample_key(latest), 1

    start_index = 0
    for index in range(len(samples) - 1, -1, -1):
        if _metrics_sample_key(samples[index]) == last_metrics_key:
            start_index = index + 1
            break
    unread = samples[start_index:]
    if not unread:
        return None, last_metrics_key, 0
    return _aggregate_metrics_samples(unread), _metrics_sample_key(unread[-1]), len(unread)


def _read_unread_metrics_from_files(
    paths: Sequence[Path],
    last_metrics_key: Optional[str],
) -> Tuple[Optional[Dict[str, object]], Optional[str], int]:
    previous_state = _decode_multi_metrics_state(last_metrics_key)
    next_state = dict(previous_state)
    per_file_metrics: List[Dict[str, object]] = []
    total_unread = 0

    for path in paths:
        samples = _read_jsonl_samples(path)
        if not samples:
            continue
        path_key = str(path)
        previous_key = previous_state.get(path_key)
        if previous_key is None:
            unread = [samples[-1]]
        else:
            start_index = 0
            for index in range(len(samples) - 1, -1, -1):
                if _metrics_sample_key(samples[index]) == previous_key:
                    start_index = index + 1
                    break
            unread = samples[start_index:]
        if not unread:
            continue
        per_file_metrics.append(_aggregate_metrics_samples(unread))
        next_state[path_key] = _metrics_sample_key(unread[-1])
        total_unread += len(unread)

    if not per_file_metrics:
        return None, last_metrics_key, 0
    return (
        _aggregate_parallel_metrics_samples(per_file_metrics, total_unread),
        _encode_multi_metrics_state(next_state),
        total_unread,
    )


def _encode_multi_metrics_state(state: Mapping[str, str]) -> str:
    return "multi:" + json.dumps(dict(state), sort_keys=True)


def _decode_multi_metrics_state(last_metrics_key: Optional[str]) -> Dict[str, str]:
    if not last_metrics_key or not last_metrics_key.startswith("multi:"):
        return {}
    try:
        value = json.loads(last_metrics_key[len("multi:"):])
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _select_metrics_files(raw_dir: str) -> List[Path]:
    directory = Path(raw_dir)
    lb_files = sorted(
        directory.glob("afd_metrics_lb.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    attn_files = sorted(
        directory.glob("afd_metrics_dp*_attn.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not attn_files:
        attn_files = sorted(
            directory.glob("*_attn.jsonl"),
            key=lambda path: path.stat().st_mtime,
        )
    if attn_files:
        return lb_files + attn_files
    if lb_files:
        return lb_files
    jsonl_files = sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    if jsonl_files:
        return [jsonl_files[-1]]
    json_files = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if json_files:
        return [json_files[-1]]
    raise FileNotFoundError(f"no metrics json/jsonl files found in {directory}")


def _read_latest_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix == ".jsonl":
        samples = _read_jsonl_samples(path)
        if not samples:
            raise ValueError(f"no valid JSON lines found in {path}")
        return samples[-1]
    with open(path, "r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"metrics JSON must be an object: {path}")
    return value


def _read_jsonl_samples(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    samples: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                samples.append(value)
    return samples


def _aggregate_metrics_samples(
    samples: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    latest = dict(samples[-1])
    count = len(samples)
    worker_type = str(latest.get("worker_type", "")).lower()
    window_s = sum(
        max(0.0, _first_float(sample, ("window_s",), default=0.0))
        for sample in samples
    )
    window_requests = sum(
        max(0.0, _first_float(sample, ("window_requests",), default=0.0))
        for sample in samples
    )
    received_requests = sum(
        max(0.0, _first_float(sample, ("window_received_requests",), default=0.0))
        for sample in samples
    )
    finished_requests = sum(
        max(0.0, _first_float(sample, ("window_finished_requests",), default=0.0))
        for sample in samples
    )
    failed_requests = sum(
        max(0.0, _first_float(sample, ("window_failed_requests",), default=0.0))
        for sample in samples
    )
    finished_output_tokens = sum(
        _finished_output_tokens(sample) for sample in samples
    )
    running_requests = max(
        0.0,
        _first_float(
            latest,
            ("running_requests", "num_running_requests", "running_reqs"),
            default=0.0,
        ),
    )
    has_running_output_shape = any(
        key in latest
        for key in ("running_output_tokens", "avg_running_output_tokens")
    )
    running_output_tokens = (
        _running_output_tokens(latest) if has_running_output_shape else 0.0
    )

    arrival_rate = (
        window_requests / window_s
        if worker_type == "lb" and window_s > 0
        else _mean_metric(
            samples,
            ("arrival_rate_rps", "arrival_rate", "request_throughput", "rps"),
            default=0.0,
        )
    )
    finished_requests_per_sec = _mean_metric(
        samples,
        ("finished_requests_per_sec", "finished_request_rate_rps"),
        default=0.0,
    )
    avg_input = _weighted_mean_metric(
        samples,
        ("avg_input_tokens", "mean_input", "input_tokens"),
        (
            "window_received_requests",
            "arrival_rate_rps",
            "arrival_rate",
            "request_throughput",
            "rps",
        ),
        default=0.0,
    )
    completed_avg_output = _weighted_mean_metric(
        samples,
        ("avg_output_tokens", "mean_output", "output_tokens"),
        (
            "window_finished_requests",
            "finished_requests_per_sec",
            "finished_request_rate_rps",
        ),
        default=0.0,
    )
    if finished_requests > 0 and finished_output_tokens > 0:
        completed_avg_output = finished_output_tokens / finished_requests
    avg_running_output = (
        running_output_tokens / running_requests
        if has_running_output_shape and running_requests > 0
        else 0.0
    )
    avg_output = (
        completed_avg_output
        if completed_avg_output > 0
        else avg_running_output
    )

    latest.update(
        {
            "arrival_rate_rps": arrival_rate,
            "finished_requests_per_sec": finished_requests_per_sec,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "completed_avg_output_tokens": completed_avg_output,
            "aggregated_window_s": window_s,
            "aggregated_window_requests": window_requests,
            "aggregated_received_requests": received_requests,
            "aggregated_finished_requests": finished_requests,
            "aggregated_failed_requests": failed_requests,
            "aggregated_finished_output_tokens": finished_output_tokens,
            "running_requests": running_requests,
            "running_output_tokens": running_output_tokens,
            "avg_running_output_tokens": avg_running_output,
            "has_running_output_shape": has_running_output_shape,
            "waiting_requests": _first_float(
                latest,
                ("waiting_requests", "num_waiting_requests", "waiting_reqs"),
                default=0.0,
            ),
            "aggregated_metrics_samples": count,
            "aggregated_start_timestamp": samples[0].get(
                "timestamp",
                samples[0].get("timestamp_s"),
            ),
            "aggregated_end_timestamp": samples[-1].get(
                "timestamp",
                samples[-1].get("timestamp_s"),
            ),
        }
    )
    return latest


def _aggregate_parallel_metrics_samples(
    samples: Sequence[Mapping[str, object]],
    raw_sample_count: int,
) -> Dict[str, object]:
    latest = dict(max(samples, key=_sample_timestamp))
    lb_samples = [
        sample for sample in samples
        if str(sample.get("worker_type", "")).lower() == "lb"
    ]
    worker_samples = [
        sample for sample in samples
        if str(sample.get("worker_type", "")).lower() != "lb"
    ]
    arrival_source_samples = lb_samples or samples
    metric_source_samples = worker_samples or samples

    arrival_rate = sum(
        _first_float(
            sample,
            ("arrival_rate_rps", "arrival_rate", "request_throughput", "rps"),
            default=0.0,
        )
        for sample in arrival_source_samples
    )
    shape_arrival_rate = sum(
        _first_float(
            sample,
            ("arrival_rate_rps", "arrival_rate", "request_throughput", "rps"),
            default=0.0,
        )
        for sample in metric_source_samples
    )
    finished_requests_per_sec = sum(
        _first_float(
            sample,
            ("finished_requests_per_sec", "finished_request_rate_rps"),
            default=0.0,
        )
        for sample in metric_source_samples
    )
    finished_requests = sum(
        _first_float(
            sample,
            ("aggregated_finished_requests", "window_finished_requests"),
            default=0.0,
        )
        for sample in metric_source_samples
    )
    failed_requests = sum(
        _first_float(
            sample,
            ("aggregated_failed_requests", "window_failed_requests"),
            default=0.0,
        )
        for sample in metric_source_samples
    )
    finished_output_tokens = sum(
        _finished_output_tokens(sample) for sample in metric_source_samples
    )
    running_requests = sum(
        _first_float(
            sample,
            ("running_requests", "num_running_requests", "running_reqs"),
            default=0.0,
        )
        for sample in metric_source_samples
    )
    running_output_tokens = sum(
        _running_output_tokens(sample)
        for sample in metric_source_samples
        if bool(sample.get("has_running_output_shape", False))
    )
    has_running_output_shape = any(
        bool(sample.get("has_running_output_shape", False))
        for sample in metric_source_samples
    )
    avg_input = _weighted_mean_metric(
        metric_source_samples,
        ("avg_input_tokens", "mean_input", "input_tokens"),
        (
            "aggregated_received_requests",
            "window_received_requests",
            "arrival_rate_rps",
            "arrival_rate",
            "request_throughput",
            "rps",
        ),
        default=0.0,
    )
    completed_avg_output = _weighted_mean_metric(
        metric_source_samples,
        ("avg_output_tokens", "mean_output", "output_tokens"),
        (
            "aggregated_finished_requests",
            "window_finished_requests",
            "finished_requests_per_sec",
            "finished_request_rate_rps",
        ),
        default=0.0,
    )
    if finished_requests > 0 and finished_output_tokens > 0:
        completed_avg_output = finished_output_tokens / finished_requests
    avg_running_output = (
        running_output_tokens / running_requests
        if has_running_output_shape and running_requests > 0
        else 0.0
    )
    avg_output = (
        completed_avg_output
        if completed_avg_output > 0
        else avg_running_output
    )

    latest.update(
        {
            "arrival_rate_rps": arrival_rate,
            "shape_arrival_rate_rps": shape_arrival_rate,
            "finished_requests_per_sec": finished_requests_per_sec,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "completed_avg_output_tokens": completed_avg_output,
            "aggregated_received_requests": sum(
                _first_float(
                    sample,
                    ("aggregated_received_requests", "window_received_requests"),
                    default=0.0,
                )
                for sample in metric_source_samples
            ),
            "aggregated_finished_requests": finished_requests,
            "aggregated_failed_requests": failed_requests,
            "aggregated_finished_output_tokens": finished_output_tokens,
            "running_requests": running_requests,
            "running_output_tokens": running_output_tokens,
            "avg_running_output_tokens": avg_running_output,
            "has_running_output_shape": has_running_output_shape,
            "waiting_requests": sum(
                _first_float(
                    sample,
                    ("waiting_requests", "num_waiting_requests", "waiting_reqs"),
                    default=0.0,
                )
                for sample in samples
            ),
            "aggregated_metrics_samples": raw_sample_count,
            "aggregated_metrics_files": len(samples),
            "aggregated_start_timestamp": min(
                _sample_start_timestamp(sample) for sample in samples
            ),
            "aggregated_end_timestamp": max(
                _sample_end_timestamp(sample) for sample in samples
            ),
        }
    )
    return latest


def _metrics_sample_key(metrics: Mapping[str, object]) -> str:
    for key in ("sample_id", "sample_index", "sequence", "seq", "timestamp", "timestamp_s"):
        value = metrics.get(key)
        if value is not None:
            return f"{key}:{value}"
    return json.dumps(metrics, sort_keys=True, default=str)


def _sample_timestamp(sample: Mapping[str, object]) -> float:
    return _first_float(sample, ("timestamp", "timestamp_s"), default=0.0)


def _sample_start_timestamp(sample: Mapping[str, object]) -> float:
    return _first_float(
        sample,
        ("aggregated_start_timestamp", "timestamp", "timestamp_s"),
        default=0.0,
    )


def _sample_end_timestamp(sample: Mapping[str, object]) -> float:
    return _first_float(
        sample,
        ("aggregated_end_timestamp", "timestamp", "timestamp_s"),
        default=0.0,
    )


def _mean_metric(
    samples: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    *,
    default: float,
    positive_only: bool = False,
) -> float:
    values = []
    for sample in samples:
        value = _first_float(sample, keys, default=float("nan"))
        if value != value:
            continue
        if positive_only and value <= 0:
            continue
        values.append(value)
    if not values:
        return default
    return sum(values) / len(values)


def _finished_output_tokens(sample: Mapping[str, object]) -> float:
    for key in (
        "aggregated_finished_output_tokens",
        "window_finished_output_tokens",
    ):
        if key in sample:
            return max(0.0, _first_float(sample, (key,), default=0.0))
    finished_requests = _first_float(
        sample,
        ("aggregated_finished_requests", "window_finished_requests"),
        default=0.0,
    )
    avg_output = _first_float(
        sample,
        ("avg_output_tokens", "mean_output", "output_tokens"),
        default=0.0,
    )
    return max(0.0, finished_requests) * max(0.0, avg_output)


def _running_output_tokens(sample: Mapping[str, object]) -> float:
    if "running_output_tokens" in sample:
        return max(
            0.0,
            _first_float(sample, ("running_output_tokens",), default=0.0),
        )
    running_requests = _first_float(
        sample,
        ("running_requests", "num_running_requests", "running_reqs"),
        default=0.0,
    )
    avg_output = _first_float(
        sample,
        ("avg_running_output_tokens",),
        default=0.0,
    )
    return max(0.0, running_requests) * max(0.0, avg_output)


def _weighted_mean_metric(
    samples: Sequence[Mapping[str, object]],
    value_keys: Sequence[str],
    weight_keys: Sequence[str],
    *,
    default: float,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    values = []
    for sample in samples:
        value = _first_float(sample, value_keys, default=float("nan"))
        if value != value or value <= 0:
            continue
        values.append(value)
        weight = _first_positive_float(sample, weight_keys, default=0.0)
        if weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight > 0:
        return weighted_sum / total_weight
    if values:
        return sum(values) / len(values)
    return default


def _workload_from_metrics(
    args,
    metrics: Mapping[str, object],
    *,
    shape_state: Optional[Dict[str, Tuple[float, float]]] = None,
    now_s: Optional[float] = None,
) -> WorkloadProfile:
    arrival_rate = _first_float(
        metrics,
        ("arrival_rate_rps", "arrival_rate", "request_throughput", "rps"),
        default=float(INITIAL_METRICS["arrival_rate_rps"]),
    )
    observed_at = time.time() if now_s is None else float(now_s)
    max_age_s = max(
        0.0,
        float(getattr(args, "workload_shape_max_age_s", 300.0)),
    )
    avg_input = _current_or_cached_shape_metric(
        metrics,
        keys=("avg_input_tokens", "mean_input", "input_tokens"),
        label="input tokens",
        state_key="input",
        shape_state=shape_state,
        observed_at=observed_at,
        max_age_s=max_age_s,
    )
    avg_output = _stable_output_shape_metric(
        metrics,
        shape_state=shape_state,
        observed_at=observed_at,
        max_age_s=max_age_s,
    )
    return WorkloadProfile(
        arrival_rate=max(arrival_rate, 1e-9),
        input_lengths=(max(1, int(round(avg_input))),),
        output_lengths=(max(1, int(round(avg_output))),),
    )


def _stable_output_shape_metric(
    metrics: Mapping[str, object],
    *,
    shape_state: Optional[Dict[str, Tuple[float, float]]],
    observed_at: float,
    max_age_s: float,
) -> float:
    completed_requests = max(
        0.0,
        _first_float(
            metrics,
            ("aggregated_finished_requests", "window_finished_requests"),
            default=0.0,
        ),
    )
    completed_avg = _first_float(
        metrics,
        ("completed_avg_output_tokens",),
        default=0.0,
    )
    if completed_avg <= 0 and completed_requests > 0:
        completed_tokens = _finished_output_tokens(metrics)
        if completed_tokens > 0:
            completed_avg = completed_tokens / completed_requests

    running_requests = max(
        0.0,
        _first_float(
            metrics,
            ("running_requests", "num_running_requests", "running_reqs"),
            default=0.0,
        ),
    )
    running_avg = _first_float(
        metrics,
        ("avg_running_output_tokens",),
        default=0.0,
    )
    if running_avg <= 0 and running_requests > 0:
        running_tokens = _running_output_tokens(metrics)
        if running_tokens > 0:
            running_avg = running_tokens / running_requests

    direct_avg = _first_float(
        metrics,
        ("avg_output_tokens", "mean_output", "output_tokens"),
        default=0.0,
    )
    cached_avg = _cached_shape_metric(
        shape_state,
        "output",
        observed_at=observed_at,
        max_age_s=max_age_s,
    )

    has_observation = False
    if completed_requests > 0 and completed_avg > 0:
        value = max(completed_avg, running_avg)
        has_observation = True
    elif running_avg > 0:
        # Running output is a right-censored lower bound, not a completed sample.
        value = max(running_avg, cached_avg)
        has_observation = True
    elif direct_avg > 0:
        value = direct_avg
        has_observation = True
    else:
        value = cached_avg

    if value <= 0:
        raise ValueError("metrics missing positive output tokens")
    if shape_state is not None and has_observation:
        shape_state["output"] = (value, observed_at)
    return value


def _current_or_cached_shape_metric(
    metrics: Mapping[str, object],
    *,
    keys: Sequence[str],
    label: str,
    state_key: str,
    shape_state: Optional[Dict[str, Tuple[float, float]]],
    observed_at: float,
    max_age_s: float,
) -> float:
    value = _first_float(metrics, keys, default=0.0)
    if value > 0:
        if shape_state is not None:
            shape_state[state_key] = (value, observed_at)
        return value
    cached_value = _cached_shape_metric(
        shape_state,
        state_key,
        observed_at=observed_at,
        max_age_s=max_age_s,
    )
    if cached_value > 0:
        return cached_value
    raise ValueError(f"metrics missing positive {label}")


def _cached_shape_metric(
    shape_state: Optional[Dict[str, Tuple[float, float]]],
    state_key: str,
    *,
    observed_at: float,
    max_age_s: float,
) -> float:
    if shape_state is None or state_key not in shape_state:
        return 0.0
    cached_value, cached_at = shape_state[state_key]
    age_s = max(0.0, observed_at - cached_at)
    if cached_value > 0 and age_s <= max_age_s:
        return cached_value
    shape_state.pop(state_key, None)
    return 0.0


def _first_float(
    values: Mapping[str, object],
    keys: Sequence[str],
    *,
    default: float,
) -> float:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _first_positive_float(
    values: Mapping[str, object],
    keys: Sequence[str],
    *,
    default: float,
) -> float:
    for key in keys:
        value = _first_float(values, (key,), default=float("nan"))
        if value == value and value > 0:
            return value
    return default


def _parse_capacity(value: Mapping[str, int] | str) -> Dict[str, int]:
    raw = json.loads(value) if isinstance(value, str) else dict(value)
    capacity = {str(hardware): int(count) for hardware, count in raw.items()}
    if any(count < 0 for count in capacity.values()):
        raise ValueError("capacity values must be non-negative")
    if sum(capacity.values()) < len(WORKER_TYPES):
        raise ValueError("capacity must have at least one GPU per worker type")
    return capacity


def _parse_initial_allocation(
    value: Mapping[str, int] | str,
    capacity: Mapping[str, int],
) -> AllocationMatrix:
    raw = json.loads(value) if isinstance(value, str) else dict(value)
    targets = {worker: int(raw.get(worker, 0)) for worker in WORKER_TYPES}
    missing = [worker for worker in WORKER_TYPES if worker not in raw]
    if missing:
        raise ValueError(f"initial allocation missing workers: {missing}")
    if any(count <= 0 for count in targets.values()):
        raise ValueError("initial allocation values must be positive")
    total = sum(targets.values())
    hardware = next(
        (name for name, count in capacity.items() if int(count) >= total),
        None,
    )
    if hardware is None:
        raise ValueError(
            f"initial allocation needs {total} GPUs on one hardware type, "
            f"capacity={dict(capacity)}"
        )
    allocation = AllocationMatrix.zeros(capacity.keys())
    for worker, count in targets.items():
        allocation.set(worker, hardware, count)
    allocation.validate(capacity)
    return allocation


def _parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("expected a comma-separated list of positive integers")
    return values


def _result_payload(
    args,
    windows: Sequence[Mapping[str, object]],
    capacity: Mapping[str, int],
    initial_allocation: AllocationMatrix,
) -> Dict[str, object]:
    return {
        "config": {
            "model_path": args.model_path,
            "capacity": dict(capacity),
            "initial_allocation": initial_allocation.values,
            "estimator": {
                "kv_transfer_bandwidth_gbps": args.kv_transfer_bandwidth_gbps,
                "activation_bandwidth_gbps": args.activation_bandwidth_gbps,
            },
            "runtime": {
                "afd_micro_batch": args.afd_micro_batch,
                "afd_sched_host": args.afd_sched_host,
                "disaggregation_ib_device": args.disaggregation_ib_device,
                "runtime_host": args.runtime_host,
                "runtime_base_port": args.runtime_base_port,
                "lb_host": args.lb_host,
                "lb_port": args.lb_port,
            },
            "autoscaling": {
                "target_utilization": args.target_utilization,
                "hysteresis": args.hysteresis,
                "decode_worker_gpu_choices": _parse_int_list(args.decode_worker_gpu_choices),
                "stability_search_max_rounds": args.stability_search_max_rounds,
                "stability_search_max_candidates": args.stability_search_max_candidates,
                "stability_search_timeout_s": args.stability_search_timeout_s,
                "scale_in_backlog_threshold": args.scale_in_backlog_threshold,
            },
        },
        "windows": list(windows),
    }


def _write_json(path: str, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, output)


def _print_step_summary(window: Mapping[str, object]) -> None:
    plan = window["scheduling_plan"]
    workload = window["workload"]
    print(
        f"[{window['index']}] {window['reschedule_decision']}: "
        f"arrival={float(workload['arrival_rate']):.4f} req/s "
        f"input={int(workload['mean_input'])} "
        f"output={int(workload['mean_output'])} "
        f"throughput={float(plan['system_throughput_req_s']):.4f} req/s "
        f"latency={float(plan['estimated_latency_s']):.4f}s "
        f"parallelism={_compact_runtime_parallelism(window)}"
    )


def _print_stability_search_summary(window: Mapping[str, object]) -> None:
    scheduling_plan = window.get("scheduling_plan", {})
    if not isinstance(scheduling_plan, Mapping):
        return
    metadata = scheduling_plan.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return
    autoscaling = metadata.get("autoscaling", {})
    if not isinstance(autoscaling, Mapping):
        return
    search = autoscaling.get("stability_validation", {})
    if not isinstance(search, Mapping):
        return

    selection = str(search.get("selection", ""))
    if not bool(search.get("triggered")) and selection == "stable_fixed_point":
        return

    path = search.get("proposal_path", ())
    compact_path = " -> ".join(
        _compact_allocation(allocation)
        for allocation in path
        if isinstance(allocation, Mapping)
    )
    selected = search.get("selected_allocation", {})
    selected_text = (
        _compact_allocation(selected) if isinstance(selected, Mapping) else "unknown"
    )
    print(
        f"[{window['index']}] stability_validation: selection={selection} "
        f"path={compact_path or 'unknown'} selected={selected_text} "
        f"throughput={float(search.get('selected_throughput_req_s', 0.0)):.4f} "
        f"required={float(search.get('required_safe_throughput_req_s', 0.0)):.4f} "
        f"candidates={int(search.get('evaluated_candidates', 0))}"
    )


def _compact_allocation(allocation: Mapping[str, object]) -> str:
    totals = []
    for worker in WORKER_TYPES:
        by_hardware = allocation.get(worker, {})
        total = (
            sum(int(value) for value in by_hardware.values())
            if isinstance(by_hardware, Mapping)
            else 0
        )
        totals.append(f"{worker}{total}")
    return "/".join(totals)


def _compact_runtime_parallelism(window: Mapping[str, object]) -> str:
    launch = window.get("runtime_launch", {})
    if not isinstance(launch, Mapping):
        return "unknown"
    metadata = launch.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return "unknown"
    specs = (
        ("pre", "prefill"),
        ("attn", "attention"),
        ("ffn", "ffn"),
    )
    return " ".join(
        f"{label}={metadata.get(prefix + '_gpus')}g/"
        f"dp{metadata.get(prefix + '_dp')}/tp{metadata.get(prefix + '_tp')}"
        for label, prefix in specs
    )


def _sleep(args) -> None:
    time.sleep(max(0.0, float(args.poll_interval_s)))


if __name__ == "__main__":
    sys.exit(main())
