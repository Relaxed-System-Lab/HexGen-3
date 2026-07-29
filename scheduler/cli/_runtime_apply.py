#!/usr/bin/env python3
"""Runtime apply helper used by the HexGen-3 live autoscaler."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ApplyRuntimeOptions:
    plan_path: str
    window: Mapping[str, object]
    gpu_ids: Optional[str] = None
    cwd: Optional[str] = None
    log_dir: str = "/tmp/hexgen3_runtime_logs"
    pid_file: str = "/tmp/hexgen3_runtime_pids.json"
    extra_env: Mapping[str, str] = field(default_factory=dict)
    lb_host: str = "127.0.0.1"
    lb_port: int = 30000
    drain_state_file: str = "/tmp/afd_runtime_state.json"
    drain_wait_timeout_s: float = 600.0
    startup_wait_s: float = 2.0
    restart_lb: bool = True
    cleanup_stale_runtime: bool = False


def apply_runtime_window(options: ApplyRuntimeOptions) -> int:
    window = options.window
    launch = window.get("runtime_launch", {})
    if not isinstance(launch, Mapping):
        raise ValueError(f"window {window.get('name')!r} has no runtime_launch")

    runtime_processes = list(launch.get("processes", []))
    if not runtime_processes:
        raise ValueError(f"window {window.get('name')!r} has no runtime processes")
    assignments = _assign_gpus(runtime_processes, options.gpu_ids)
    extra_env = dict(options.extra_env or {})
    cwd_override = str(Path(options.cwd).resolve()) if options.cwd else None
    log_dir = Path(options.log_dir)
    kept_lb_processes = (
        _kept_existing_processes(options.pid_file, keep_roles={"lb"})
        if not options.restart_lb
        else []
    )
    launch_lb = options.restart_lb or not kept_lb_processes
    processes = runtime_processes
    if launch_lb:
        processes = runtime_processes + [_build_lb_process(launch, options)]

    _stop_existing(
        options.pid_file,
        keep_roles={"lb"} if kept_lb_processes else set(),
    )
    if options.cleanup_stale_runtime:
        _cleanup_stale_runtime_processes(include_lb=launch_lb)
    else:
        print("skip stale runtime process scan; pass --cleanup-stale-runtime for crash cleanup")
    _cleanup_rdma_cm_users()

    log_dir.mkdir(parents=True, exist_ok=True)
    launched = [
        _launch_one_process(
            index=index,
            process=process,
            gpu_ids=assignments.get(str(process["name"]), []),
            extra_env=extra_env,
            cwd_override=cwd_override,
            log_dir=log_dir,
        )
        for index, process in enumerate(processes)
    ]
    _write_pid_file(
        options.pid_file,
        plan_path=str(Path(options.plan_path).resolve()),
        window=window,
        launched=list(kept_lb_processes) + launched,
        log_dir=str(log_dir.resolve()),
    )

    if options.startup_wait_s > 0:
        time.sleep(options.startup_wait_s)
    failed = _check_immediate_exits(launched)
    _print_launch_result(launched, failed, options.pid_file)
    return 1 if failed else 0


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _build_lb_process(
    launch: Mapping[str, object],
    args,
) -> Dict[str, object]:
    processes = list(launch.get("processes", []))
    prefill = next(
        process for process in processes if str(process.get("role")) == "prefill"
    )
    attention = next(
        process for process in processes if str(process.get("role")) == "attention"
    )

    command = [
        "python3",
        "-m",
        "sglang.srt.disaggregation.mini_lb",
        "--prefill",
        _server_url(prefill),
        "--decode",
        _server_url(attention),
        "--host",
        args.lb_host,
        "--port",
        str(args.lb_port),
        "--drain-state-file",
        args.drain_state_file,
        "--drain-wait-timeout-s",
        str(args.drain_wait_timeout_s),
    ]
    return {
        "name": "mini-lb",
        "role": "lb",
        "command": command,
        "env": {},
        "metadata": {"gpus": 0},
    }


def _server_url(process: Mapping[str, object]) -> str:
    command = [str(part) for part in process.get("command", [])]
    host = _command_option(command, "--host", "127.0.0.1")
    port = _command_option(command, "--port", None)
    if port is None:
        raise ValueError(f"process {process.get('name')} has no --port")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _command_option(
    command: Sequence[str],
    option: str,
    default: Optional[str],
) -> Optional[str]:
    try:
        index = command.index(option)
    except ValueError:
        return default
    if index + 1 >= len(command):
        return default
    return command[index + 1]


def _assign_gpus(
    processes: Sequence[Mapping[str, object]],
    raw_gpu_ids: Optional[str],
) -> Dict[str, List[str]]:
    total_gpus = sum(_process_gpus(process) for process in processes)
    gpu_ids = (
        [item.strip() for item in raw_gpu_ids.split(",") if item.strip()]
        if raw_gpu_ids
        else [str(index) for index in range(total_gpus)]
    )
    if len(gpu_ids) < total_gpus:
        raise ValueError(f"need {total_gpus} GPU ids, got {len(gpu_ids)}")
    offset = 0
    assignments: Dict[str, List[str]] = {}
    for process in processes:
        count = _process_gpus(process)
        name = str(process["name"])
        assignments[name] = gpu_ids[offset:offset + count]
        offset += count
    return assignments


def _process_gpus(process: Mapping[str, object]) -> int:
    metadata = process.get("metadata", {})
    if isinstance(metadata, Mapping):
        return max(0, int(metadata.get("gpus", 1)))
    return 1


def _effective_env(
    process: Mapping[str, object],
    gpu_ids: Sequence[str],
    extra_env: Mapping[str, str],
) -> Dict[str, str]:
    env = dict(process.get("env", {}))
    env.update(extra_env)
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return env


def _kept_existing_processes(
    pid_file: str,
    *,
    keep_roles: set[str],
) -> List[Mapping[str, object]]:
    if not keep_roles:
        return []
    path = Path(pid_file)
    if not path.exists():
        return []
    data = _load_json(str(path))
    processes = data.get("processes", [])
    if not isinstance(processes, list):
        return []
    kept = [
        process
        for process in processes
        if isinstance(process, Mapping)
        and str(process.get("role", "")) in keep_roles
        and _pid_exists(int(process["pid"]))
    ]
    if kept:
        print(
            "keeping existing processes: "
            + ", ".join(str(process.get("name", process.get("pid"))) for process in kept)
        )
    return kept


def _stop_existing(pid_file: str, *, keep_roles: set[str]) -> None:
    path = Path(pid_file)
    if not path.exists():
        print(f"no existing pid file: {pid_file}")
        return
    data = _load_json(str(path))
    processes = data.get("processes", [])
    if not isinstance(processes, list):
        return
    stop_processes = [
        process
        for process in processes
        if isinstance(process, Mapping)
        and str(process.get("role", "")) not in keep_roles
    ]
    if keep_roles:
        kept_names = [
            str(process.get("name", process.get("pid")))
            for process in processes
            if isinstance(process, Mapping)
            and str(process.get("role", "")) in keep_roles
        ]
        if kept_names:
            print(f"keeping existing roles {sorted(keep_roles)}: {', '.join(kept_names)}")
    print(f"stopping {len(stop_processes)} existing processes from {pid_file}")
    targets = [
        (int(process["pid"]), f"{process.get('name', process['pid'])} pgid/pid={process['pid']}")
        for process in stop_processes
    ]
    _terminate_process_targets(
        targets,
        process_group=True,
        grace_s=2.0,
        already_exited=True,
    )


def _cleanup_stale_runtime_processes(*, include_lb: bool) -> None:
    patterns = [
        "sglang.launch_server",
        "torch/_inductor/compile_worker",
        "sglang::",
    ]
    if include_lb:
        patterns.append("sglang.srt.disaggregation.mini_lb")
    current_pids = {os.getpid(), os.getppid()}
    pid_to_pattern: Dict[int, str] = {}
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            print("pgrep not found; skip stale runtime cleanup")
            return
        if result.returncode not in (0, 1):
            print(f"pgrep returned {result.returncode} for {pattern}; skip")
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid not in current_pids:
                pid_to_pattern.setdefault(pid, pattern)

    if not pid_to_pattern:
        print("no stale HexGen-3 runtime processes found")
        return

    pids = sorted(pid_to_pattern)
    print(f"cleaning {len(pids)} stale HexGen-3 runtime processes: {pids}")
    targets = [(pid, f"stale runtime pid={pid} pattern={pid_to_pattern[pid]}") for pid in pids]
    _terminate_process_targets(targets, process_group=True, grace_s=2.0)


def _cleanup_rdma_cm_users() -> None:
    device = "/dev/infiniband/rdma_cm"
    print(f"checking stale RDMA users: {device}")
    try:
        result = subprocess.run(
            ["lsof", "-t", device],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        print("  lsof not found; skip RDMA cleanup")
        return
    if result.returncode not in (0, 1):
        print(f"  lsof returned {result.returncode}; skip RDMA cleanup")
        return

    current_pids = {os.getpid(), os.getppid()}
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid not in current_pids:
            pids.append(pid)

    unique_pids = sorted(set(pids))
    if not unique_pids:
        print("  no stale RDMA users found")
        return

    print(f"  cleaning {len(unique_pids)} RDMA user processes: {unique_pids}")
    targets = [(pid, f"rdma_cm pid={pid}") for pid in unique_pids]
    _terminate_process_targets(targets, process_group=False, grace_s=1.0)


def _terminate_process_targets(
    targets: Sequence[Tuple[int, str]],
    *,
    process_group: bool,
    grace_s: float,
    already_exited: bool = False,
) -> None:
    for pid, description in targets:
        if _send_signal(pid, signal.SIGTERM, process_group=process_group):
            print(f"  SIGTERM {description}")
        elif already_exited:
            print(f"  already exited {description}")
    time.sleep(grace_s)
    for pid, description in targets:
        if not _pid_exists(pid):
            continue
        if _send_signal(pid, signal.SIGKILL, process_group=process_group):
            print(f"  SIGKILL {description}")


def _send_signal(pid: int, sig: signal.Signals, *, process_group: bool) -> bool:
    try:
        if process_group:
            try:
                os.killpg(pid, sig)
                return True
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _launch_one_process(
    *,
    index: int,
    process: Mapping[str, object],
    gpu_ids: Sequence[str],
    extra_env: Mapping[str, str],
    cwd_override: Optional[str],
    log_dir: Path,
) -> Dict[str, object]:
    name = str(process["name"])
    role = str(process.get("role", "unknown"))
    command = [str(part) for part in process["command"]]
    cwd = cwd_override or process.get("cwd") or os.getcwd()
    env = os.environ.copy()
    env.update(_effective_env(process, gpu_ids, extra_env))
    stdout_path = log_dir / f"{index:02d}_{name}.out"
    stderr_path = log_dir / f"{index:02d}_{name}.err"
    stdout = open(stdout_path, "ab")
    stderr = open(stderr_path, "ab")
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    finally:
        stdout.close()
        stderr.close()
    launched = {
        "name": name,
        "role": role,
        "pid": proc.pid,
        "command": command,
        "cwd": cwd,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    print(
        f"launched {name:<14} role={role:<9} pid={proc.pid:<8} "
        f"gpus={launched['cuda_visible_devices'] or '-'} "
        f"logs={stdout_path},{stderr_path}"
    )
    return launched


def _write_pid_file(
    pid_file: str,
    *,
    plan_path: str,
    window: Mapping[str, object],
    launched: Sequence[Mapping[str, object]],
    log_dir: str,
) -> None:
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "plan": plan_path,
        "window": {"index": window.get("index"), "name": window.get("name")},
        "log_dir": log_dir,
        "processes": list(launched),
    }
    path = Path(pid_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _check_immediate_exits(
    launched: Sequence[Mapping[str, object]],
) -> List[Tuple[Mapping[str, object], int]]:
    failed = []
    for process in launched:
        pid = int(process["pid"])
        if not _pid_exists(pid):
            failed.append((process, -1))
    return failed


def _print_launch_result(
    launched: Sequence[Mapping[str, object]],
    failed: Sequence[Tuple[Mapping[str, object], int]],
    pid_file: str,
) -> None:
    print("-" * 96)
    print(f"pid file: {pid_file}")
    if not failed:
        print(f"launch status: {len(launched)} processes still running after startup check")
        return
    print(f"launch status: {len(failed)} process(es) exited during startup check")
    for process, return_code in failed:
        print(
            f"  {process['name']} pid={process['pid']} rc={return_code} "
            f"stderr={process['stderr']}"
        )
