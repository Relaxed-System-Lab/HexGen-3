"""Runtime launch specs for HexGen-3 AFD deployments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from .types import DeploymentPlan, ParallelismStrategy, WORKER_TYPES


RUNTIME_NAME = "sglang-hexgen3-stepmesh"


@dataclass(frozen=True)
class RuntimeProcessSpec:
    """One process the runtime launcher can start."""

    name: str
    role: str
    command: Tuple[str, ...]
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "role": self.role,
            "command": list(self.command),
            "env": dict(self.env),
            "metadata": dict(self.metadata),
        }
        if self.cwd is not None:
            payload["cwd"] = self.cwd
        return payload


@dataclass(frozen=True)
class RuntimeLaunchSpec:
    """A scheduler-produced description of the HexGen-3 runtime processes."""

    runtime: str
    processes: Tuple[RuntimeProcessSpec, ...]
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "runtime": self.runtime,
            "processes": [process.as_dict() for process in self.processes],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AFDDeploymentSpec:
    """HexGen-3 AFD deployment shape derived from a scheduler plan."""

    model_path: str
    prefill_gpus: int
    attention_gpus: int
    ffn_gpus: int
    prefill_dp: int = 1
    prefill_tp: int = 1
    attention_dp: int = 1
    attention_tp: int = 1
    ffn_dp: int = 1
    ffn_tp: int = 1
    afd_micro_batch: int = 2
    disaggregation_ib_device: str = "mlx5_bond_1"
    disaggregation_bootstrap_port: int = 9001
    host: str = "127.0.0.1"
    prefill_port: int = 30001
    attention_port: int = 30002
    ffn_port: int = 30003
    chunked_prefill_size: int = 65536
    max_prefill_tokens: int = 65536
    prefill_bootstrap_timeout_s: int = 1200
    afd_sched_host: str = "127.0.0.1"
    dmlc_ps_root_uri: Optional[str] = None
    dmlc_node_host: Optional[str] = None
    mlc_interface: str = "bond2"
    ffn_mem_fraction_static: float = 0.75
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for field_name in ("prefill_gpus", "attention_gpus", "ffn_gpus"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "prefill_dp",
            "prefill_tp",
            "attention_dp",
            "attention_tp",
            "ffn_dp",
            "ffn_tp",
            "afd_micro_batch",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        _validate_stage_shape("prefill", self.prefill_gpus, self.prefill_dp, self.prefill_tp)
        _validate_stage_shape("attention", self.attention_gpus, self.attention_dp, self.attention_tp)
        _validate_stage_shape("ffn", self.ffn_gpus, self.ffn_dp, self.ffn_tp)
        if self.attention_dp != self.ffn_dp:
            raise ValueError(
                "attention_dp must equal ffn_dp for HexGen-3 AFD communication: "
                f"{self.attention_dp} != {self.ffn_dp}"
            )

    def as_dict(self) -> Dict[str, object]:
        return {
            "model_path": self.model_path,
            "prefill_gpus": self.prefill_gpus,
            "attention_gpus": self.attention_gpus,
            "ffn_gpus": self.ffn_gpus,
            "prefill_dp": self.prefill_dp,
            "prefill_tp": self.prefill_tp,
            "attention_dp": self.attention_dp,
            "attention_tp": self.attention_tp,
            "ffn_dp": self.ffn_dp,
            "ffn_tp": self.ffn_tp,
            "afd_micro_batch": self.afd_micro_batch,
            "disaggregation_ib_device": self.disaggregation_ib_device,
            "disaggregation_bootstrap_port": self.disaggregation_bootstrap_port,
            "host": self.host,
            "prefill_port": self.prefill_port,
            "attention_port": self.attention_port,
            "ffn_port": self.ffn_port,
            "chunked_prefill_size": self.chunked_prefill_size,
            "max_prefill_tokens": self.max_prefill_tokens,
            "prefill_bootstrap_timeout_s": self.prefill_bootstrap_timeout_s,
            "afd_sched_host": self.afd_sched_host,
            "dmlc_ps_root_uri": self.dmlc_ps_root_uri,
            "dmlc_node_host": self.dmlc_node_host,
            "mlc_interface": self.mlc_interface,
            "ffn_mem_fraction_static": self.ffn_mem_fraction_static,
            "metadata": dict(self.metadata),
        }


def plan_to_afd_deployment_spec(
    plan: DeploymentPlan,
    *,
    model_path: str,
    afd_micro_batch: int = 2,
    disaggregation_ib_device: str = "mlx5_bond_1",
    disaggregation_bootstrap_port: int = 9001,
    host: str = "127.0.0.1",
    base_port: int = 30001,
    afd_sched_host: str = "127.0.0.1",
    dmlc_ps_root_uri: Optional[str] = None,
    dmlc_node_host: Optional[str] = None,
    mlc_interface: str = "bond2",
    ffn_mem_fraction_static: float = 0.75,
) -> AFDDeploymentSpec:
    """Convert a scheduler plan into the simple HexGen-3 AFD runtime shape."""

    worker_gpus = {
        worker: sum(plan.allocation.values.get(worker, {}).values())
        for worker in WORKER_TYPES
    }
    prefill_strategy = _single_uniform_strategy(plan, "pre")
    attention_strategy = _single_uniform_strategy(plan, "attn")
    ffn_strategy = _single_uniform_strategy(plan, "ffn")

    return AFDDeploymentSpec(
        model_path=model_path,
        prefill_gpus=worker_gpus["pre"],
        attention_gpus=worker_gpus["attn"],
        ffn_gpus=worker_gpus["ffn"],
        prefill_dp=prefill_strategy.dp,
        prefill_tp=prefill_strategy.tp,
        attention_dp=attention_strategy.dp,
        attention_tp=attention_strategy.tp,
        ffn_dp=ffn_strategy.dp,
        ffn_tp=ffn_strategy.tp,
        afd_micro_batch=afd_micro_batch,
        disaggregation_ib_device=disaggregation_ib_device,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        host=host,
        prefill_port=base_port,
        attention_port=base_port + 1,
        ffn_port=base_port + 2,
        afd_sched_host=afd_sched_host,
        dmlc_ps_root_uri=dmlc_ps_root_uri or afd_sched_host,
        dmlc_node_host=dmlc_node_host or afd_sched_host,
        mlc_interface=mlc_interface,
        ffn_mem_fraction_static=ffn_mem_fraction_static,
        metadata={
            "throughput": dict(plan.throughput.by_worker),
            "estimated_latency_s": plan.estimated_latency_s,
            "cost_per_hour": plan.cost_per_hour,
        },
    )


def afd_deployment_spec_to_runtime_launch_spec(
    spec: AFDDeploymentSpec,
    *,
    python_executable: str = "python",
    cwd: Optional[str] = None,
) -> RuntimeLaunchSpec:
    """Render launch commands close to the hand-run HexGen-3 command form."""

    def process(
        *,
        name: str,
        role: str,
        mode: str,
        port: int,
        gpus: int,
        dp: int,
        tp: int,
        extra_args: Sequence[str],
        env: Optional[Dict[str, str]] = None,
    ) -> RuntimeProcessSpec:
        return RuntimeProcessSpec(
            name=name,
            role=role,
            command=tuple(
                _launch_server_command(
                    spec,
                    python_executable=python_executable,
                    disaggregation_mode=mode,
                    port=port,
                    extra_args=extra_args,
                    dp=dp,
                    tp=tp,
                )
            ),
            env=env or {},
            cwd=cwd,
            metadata={"gpus": gpus},
        )

    processes = (
        process(
            name="prefill-0",
            role="prefill",
            mode="prefill",
            port=spec.prefill_port,
            gpus=spec.prefill_gpus,
            dp=spec.prefill_dp,
            tp=spec.prefill_tp,
            extra_args=[
                "--disaggregation-ib-device",
                spec.disaggregation_ib_device,
                "--disaggregation-bootstrap-port",
                str(spec.disaggregation_bootstrap_port),
                "--host",
                spec.host,
            ],
            env={
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": str(
                    spec.prefill_bootstrap_timeout_s
                )
            },
        ),
        process(
            name="decode-attn-0",
            role="attention",
            mode="decode",
            port=spec.attention_port,
            gpus=spec.attention_gpus,
            dp=spec.attention_dp,
            tp=spec.attention_tp,
            extra_args=[
                "--afd-perspective",
                "attn",
                "--disaggregation-ib-device",
                spec.disaggregation_ib_device,
                "--disaggregation-bootstrap-port",
                str(spec.disaggregation_bootstrap_port),
                "--afd-mirco-batch",
                str(spec.afd_micro_batch),
            ],
            env=_decode_env(spec),
        ),
        process(
            name="decode-ffn-0",
            role="ffn",
            mode="null",
            port=spec.ffn_port,
            gpus=spec.ffn_gpus,
            dp=spec.ffn_dp,
            tp=spec.ffn_tp,
            extra_args=[
                "--skip-server-warmup",
                "--afd-perspective",
                "ffn",
                "--afd-mirco-batch",
                str(spec.afd_micro_batch),
                "--mem-fraction-static",
                str(spec.ffn_mem_fraction_static),
            ],
            env=_decode_env(spec),
        ),
    )
    return RuntimeLaunchSpec(
        runtime=RUNTIME_NAME,
        processes=processes,
        metadata=spec.as_dict(),
    )


def _launch_server_command(
    spec: AFDDeploymentSpec,
    *,
    python_executable: str,
    disaggregation_mode: str,
    port: int,
    extra_args: Sequence[str],
    tp: int = 1,
    dp: int = 1,
) -> Tuple[str, ...]:
    command = [
        python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.model_path,
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--disaggregation-mode",
        disaggregation_mode,
        *extra_args,
        "--port",
        str(port),
        "--chunked-prefill-size",
        str(spec.chunked_prefill_size),
        "--max-prefill-tokens",
        str(spec.max_prefill_tokens),
    ]
    if tp > 1:
        command.extend(["--tp", str(tp)])
    if dp > 1:
        command.extend(["--dp", str(dp)])
    return tuple(command)


def _decode_env(spec: AFDDeploymentSpec) -> Dict[str, str]:
    return {
        "AFD_SCHED_HOST": spec.afd_sched_host,
        "DMLC_PS_ROOT_URI": spec.dmlc_ps_root_uri or spec.afd_sched_host,
        "DMLC_NODE_HOST": spec.dmlc_node_host or spec.afd_sched_host,
        "MLC_INTERFACE": spec.mlc_interface,
    }


def _single_uniform_strategy(plan: DeploymentPlan, worker: str) -> ParallelismStrategy:
    strategies = [
        strategy
        for (slice_worker, _), strategy in sorted(plan.parallelism.items())
        if slice_worker == worker
    ]
    if len(strategies) != 1:
        raise ValueError(
            f"expected exactly one {worker} strategy for single-process launch, "
            f"got {len(strategies)}"
        )
    strategy = strategies[0]
    if not strategy.is_uniform:
        raise ValueError(
            f"{worker} strategy is non-uniform and cannot be represented by a "
            "single sglang.launch_server command"
        )
    expected_gpus = sum(plan.allocation.values.get(worker, {}).values())
    _validate_stage_shape(worker, expected_gpus, strategy.dp, strategy.tp)
    return strategy


def _validate_stage_shape(stage: str, gpus: int, dp: int, tp: int) -> None:
    if int(dp) * int(tp) != int(gpus):
        raise ValueError(
            f"{stage} parallelism does not use all allocated GPUs: "
            f"dp={dp} tp={tp} gpus={gpus}"
        )
