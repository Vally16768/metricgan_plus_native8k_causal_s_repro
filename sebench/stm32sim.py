from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch


STUDENT_WEIGHT_TARGET_BYTES = 512 * 1024
STUDENT_WEIGHT_IDEAL_BYTES = 256 * 1024
STUDENT_SRAM_TARGET_BYTES = 256 * 1024
STUDENT_COMPUTE_TARGET_MS = 5.0
STUDENT_POWER_TARGET_MW = 50.0
PRODUCT_LOOKAHEAD_TARGET_MS = 80.0

DEFAULT_MCU_SHORTLIST_PROFILES = (
    "stm32u5_low_power_rt",
    "nrf54h20_low_power_rt",
    "apollo4_blue_plus_low_power_rt",
    "alif_ensemble_e3_ai_audio_rt",
    "imx_rt700_ai_audio_rt",
    "alif_ensemble_e6_ai_audio_rt",
    "stm32n6_ai_audio_rt",
    "ra8p1_ai_audio_rt",
)
DEFAULT_MCU_REFERENCE_PROFILES = (
    "ra8d1_audio_reference",
    "stm32l4x6_internal_only_rt",
)


@dataclass(frozen=True)
class MCUProfile:
    name: str
    vendor: str
    description: str
    target_class: str
    cpu_hz: int
    sample_rate: int
    window_samples: int
    hop_samples: int
    fft_size: int
    erb_bands: int
    context_frames: int
    flash_budget_ai: int
    sram_peak_budget: int
    compute_budget_ms: float
    power_budget_mw: float
    student_weight_target_bytes: int
    student_sram_target_bytes: int
    student_compute_target_ms: float
    supports_npu: bool
    idle_power_mw: float
    active_cpu_power_mw: float
    active_npu_power_mw: float
    cycles_rfft512: int
    cycles_irfft512: int
    cycles_erb_project: int
    cycles_erb_expand: int
    cycles_window_ola: int
    cycles_fc_int8_per_mac: float
    cycles_depthwise_conv1d_int8_per_mac: float
    cycles_pointwise_conv1d_int8_per_mac: float
    cycles_lstm_int8_per_mac: float
    cycles_eltwise_per_element: float
    cycles_lookup_per_element: float


PROFILES: dict[str, MCUProfile] = {
    "stm32l4x6_internal_only_rt": MCUProfile(
        name="stm32l4x6_internal_only_rt",
        vendor="ST",
        description="STM32L4x6/L4A6 internal-only reference, 80 MHz, strict real-time.",
        target_class="reference",
        cpu_hz=80_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=393_216,
        sram_peak_budget=131_072,
        compute_budget_ms=8.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=False,
        idle_power_mw=8.0,
        active_cpu_power_mw=32.0,
        active_npu_power_mw=32.0,
        cycles_rfft512=18_000,
        cycles_irfft512=18_000,
        cycles_erb_project=6_000,
        cycles_erb_expand=6_000,
        cycles_window_ola=3_000,
        cycles_fc_int8_per_mac=2.0,
        cycles_depthwise_conv1d_int8_per_mac=2.5,
        cycles_pointwise_conv1d_int8_per_mac=2.0,
        cycles_lstm_int8_per_mac=2.8,
        cycles_eltwise_per_element=0.35,
        cycles_lookup_per_element=0.5,
    ),
    "stm32u5_low_power_rt": MCUProfile(
        name="stm32u5_low_power_rt",
        vendor="ST",
        description="STM32U5 ultra-low-power shortlist profile for always-on audio.",
        target_class="low_power",
        cpu_hz=160_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=4_194_304,
        sram_peak_budget=3_145_728,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=False,
        idle_power_mw=6.0,
        active_cpu_power_mw=26.0,
        active_npu_power_mw=26.0,
        cycles_rfft512=14_000,
        cycles_irfft512=14_000,
        cycles_erb_project=4_800,
        cycles_erb_expand=4_800,
        cycles_window_ola=2_400,
        cycles_fc_int8_per_mac=1.4,
        cycles_depthwise_conv1d_int8_per_mac=1.7,
        cycles_pointwise_conv1d_int8_per_mac=1.4,
        cycles_lstm_int8_per_mac=1.9,
        cycles_eltwise_per_element=0.28,
        cycles_lookup_per_element=0.38,
    ),
    "nrf54h20_low_power_rt": MCUProfile(
        name="nrf54h20_low_power_rt",
        vendor="Nordic",
        description="nRF54H20 low-power shortlist profile for wireless audio endpoints.",
        target_class="low_power",
        cpu_hz=320_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=2_097_152,
        sram_peak_budget=1_048_576,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=False,
        idle_power_mw=7.0,
        active_cpu_power_mw=35.0,
        active_npu_power_mw=35.0,
        cycles_rfft512=10_500,
        cycles_irfft512=10_500,
        cycles_erb_project=3_200,
        cycles_erb_expand=3_200,
        cycles_window_ola=1_600,
        cycles_fc_int8_per_mac=1.1,
        cycles_depthwise_conv1d_int8_per_mac=1.35,
        cycles_pointwise_conv1d_int8_per_mac=1.1,
        cycles_lstm_int8_per_mac=1.4,
        cycles_eltwise_per_element=0.22,
        cycles_lookup_per_element=0.30,
    ),
    "apollo4_blue_plus_low_power_rt": MCUProfile(
        name="apollo4_blue_plus_low_power_rt",
        vendor="Ambiq",
        description="Apollo4 Blue Plus ultra-low-power shortlist profile for always-on voice endpoints.",
        target_class="low_power",
        cpu_hz=192_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=2_097_152,
        sram_peak_budget=2_883_584,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=False,
        idle_power_mw=4.0,
        active_cpu_power_mw=22.0,
        active_npu_power_mw=22.0,
        cycles_rfft512=12_500,
        cycles_irfft512=12_500,
        cycles_erb_project=3_900,
        cycles_erb_expand=3_900,
        cycles_window_ola=1_900,
        cycles_fc_int8_per_mac=1.25,
        cycles_depthwise_conv1d_int8_per_mac=1.55,
        cycles_pointwise_conv1d_int8_per_mac=1.25,
        cycles_lstm_int8_per_mac=1.6,
        cycles_eltwise_per_element=0.24,
        cycles_lookup_per_element=0.33,
    ),
    "alif_ensemble_e3_ai_audio_rt": MCUProfile(
        name="alif_ensemble_e3_ai_audio_rt",
        vendor="Alif",
        description="Alif Ensemble E3 shortlist profile with dual Cortex-M55 and dual Ethos-U55.",
        target_class="performance",
        cpu_hz=400_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=5_767_168,
        sram_peak_budget=14_155_776,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=True,
        idle_power_mw=18.0,
        active_cpu_power_mw=72.0,
        active_npu_power_mw=60.0,
        cycles_rfft512=7_600,
        cycles_irfft512=7_600,
        cycles_erb_project=2_200,
        cycles_erb_expand=2_200,
        cycles_window_ola=1_100,
        cycles_fc_int8_per_mac=0.18,
        cycles_depthwise_conv1d_int8_per_mac=0.26,
        cycles_pointwise_conv1d_int8_per_mac=0.18,
        cycles_lstm_int8_per_mac=0.20,
        cycles_eltwise_per_element=0.12,
        cycles_lookup_per_element=0.16,
    ),
    "imx_rt700_ai_audio_rt": MCUProfile(
        name="imx_rt700_ai_audio_rt",
        vendor="NXP",
        description="i.MX RT700 audio/AI shortlist profile, on-chip SRAM rich, NPU-assisted.",
        target_class="performance",
        cpu_hz=325_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=2_097_152,
        sram_peak_budget=7_864_320,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=True,
        idle_power_mw=22.0,
        active_cpu_power_mw=92.0,
        active_npu_power_mw=68.0,
        cycles_rfft512=8_500,
        cycles_irfft512=8_500,
        cycles_erb_project=2_500,
        cycles_erb_expand=2_500,
        cycles_window_ola=1_200,
        cycles_fc_int8_per_mac=0.25,
        cycles_depthwise_conv1d_int8_per_mac=0.35,
        cycles_pointwise_conv1d_int8_per_mac=0.25,
        cycles_lstm_int8_per_mac=0.28,
        cycles_eltwise_per_element=0.15,
        cycles_lookup_per_element=0.2,
    ),
    "alif_ensemble_e6_ai_audio_rt": MCUProfile(
        name="alif_ensemble_e6_ai_audio_rt",
        vendor="Alif",
        description="Alif Ensemble E6 shortlist profile with dual Cortex-M55 and Ethos-U85/U55 NPUs.",
        target_class="performance",
        cpu_hz=400_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=5_767_168,
        sram_peak_budget=10_223_616,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=True,
        idle_power_mw=16.0,
        active_cpu_power_mw=64.0,
        active_npu_power_mw=82.0,
        cycles_rfft512=6_900,
        cycles_irfft512=6_900,
        cycles_erb_project=1_900,
        cycles_erb_expand=1_900,
        cycles_window_ola=900,
        cycles_fc_int8_per_mac=0.10,
        cycles_depthwise_conv1d_int8_per_mac=0.14,
        cycles_pointwise_conv1d_int8_per_mac=0.10,
        cycles_lstm_int8_per_mac=0.11,
        cycles_eltwise_per_element=0.08,
        cycles_lookup_per_element=0.12,
    ),
    "stm32n6_ai_audio_rt": MCUProfile(
        name="stm32n6_ai_audio_rt",
        vendor="ST",
        description="STM32N6 shortlist profile with NPU and large SRAM.",
        target_class="performance",
        cpu_hz=800_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=2_097_152,
        sram_peak_budget=4_194_304,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=True,
        idle_power_mw=30.0,
        active_cpu_power_mw=160.0,
        active_npu_power_mw=115.0,
        cycles_rfft512=7_000,
        cycles_irfft512=7_000,
        cycles_erb_project=2_000,
        cycles_erb_expand=2_000,
        cycles_window_ola=1_000,
        cycles_fc_int8_per_mac=0.18,
        cycles_depthwise_conv1d_int8_per_mac=0.24,
        cycles_pointwise_conv1d_int8_per_mac=0.18,
        cycles_lstm_int8_per_mac=0.18,
        cycles_eltwise_per_element=0.12,
        cycles_lookup_per_element=0.15,
    ),
    "ra8p1_ai_audio_rt": MCUProfile(
        name="ra8p1_ai_audio_rt",
        vendor="Renesas",
        description="Renesas RA8P1 shortlist profile, AI-capable with moderate SRAM.",
        target_class="performance",
        cpu_hz=1_000_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=8_388_608,
        sram_peak_budget=2_097_152,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=True,
        idle_power_mw=28.0,
        active_cpu_power_mw=145.0,
        active_npu_power_mw=105.0,
        cycles_rfft512=7_500,
        cycles_irfft512=7_500,
        cycles_erb_project=2_200,
        cycles_erb_expand=2_200,
        cycles_window_ola=1_100,
        cycles_fc_int8_per_mac=0.20,
        cycles_depthwise_conv1d_int8_per_mac=0.28,
        cycles_pointwise_conv1d_int8_per_mac=0.20,
        cycles_lstm_int8_per_mac=0.20,
        cycles_eltwise_per_element=0.13,
        cycles_lookup_per_element=0.18,
    ),
    "ra8d1_audio_reference": MCUProfile(
        name="ra8d1_audio_reference",
        vendor="Renesas",
        description="Renesas RA8D1 reference profile, tighter SRAM and no strong AI target.",
        target_class="reference",
        cpu_hz=480_000_000,
        sample_rate=16_000,
        window_samples=320,
        hop_samples=160,
        fft_size=512,
        erb_bands=32,
        context_frames=5,
        flash_budget_ai=2_097_152,
        sram_peak_budget=1_048_576,
        compute_budget_ms=10.0,
        power_budget_mw=STUDENT_POWER_TARGET_MW,
        student_weight_target_bytes=STUDENT_WEIGHT_TARGET_BYTES,
        student_sram_target_bytes=STUDENT_SRAM_TARGET_BYTES,
        student_compute_target_ms=STUDENT_COMPUTE_TARGET_MS,
        supports_npu=False,
        idle_power_mw=18.0,
        active_cpu_power_mw=54.0,
        active_npu_power_mw=54.0,
        cycles_rfft512=11_000,
        cycles_irfft512=11_000,
        cycles_erb_project=3_500,
        cycles_erb_expand=3_500,
        cycles_window_ola=1_800,
        cycles_fc_int8_per_mac=1.5,
        cycles_depthwise_conv1d_int8_per_mac=1.75,
        cycles_pointwise_conv1d_int8_per_mac=1.5,
        cycles_lstm_int8_per_mac=1.25,
        cycles_eltwise_per_element=0.25,
        cycles_lookup_per_element=0.35,
    ),
}


def profile_from_name(name: str) -> MCUProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown MCU profile: {name}")
    return PROFILES[name]


def parse_profile_names(value: str | Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        return tuple(default)
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def metricgan_plus_reference_arch() -> dict[str, Any]:
    return {
        "arch": "metricgan_plus",
        "sample_rate": 16000,
        "n_fft": 512,
        "hop_length": 160,
        "win_length": 320,
        "feature_bins": 257,
        "hidden_size": 200,
        "num_layers": 2,
        "bidirectional": True,
        "linear_dims": [400, 300, 257],
        "sequence_frames": 100,
        "non_causal": True,
        "lookahead_ms": 500.0,
        "parameter_count_reference": 1_895_514,
        "weight_params_reference": 1_894_957,
        "bias_params_reference": 557,
    }


def _is_metricgan_arch(name: str) -> bool:
    return name in {"metricgan_plus", "metricgan_plus_native8k"} or name.startswith("metricgan_plus_native8k_causal_")


def _arch_audio_config(arch: dict[str, Any], profile: MCUProfile) -> dict[str, int]:
    sample_rate = int(arch.get("sample_rate") or profile.sample_rate)
    n_fft = int(arch.get("n_fft") or profile.fft_size)
    hop_length = int(arch.get("hop_length") or profile.hop_samples)
    win_length = int(arch.get("win_length") or profile.window_samples)
    erb_bands = int(arch.get("erb_bands") or profile.erb_bands)
    return {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "erb_bands": erb_bands,
    }


def _unwrap_base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "base_model", model)


def _safe_model_arch(model: torch.nn.Module) -> dict[str, Any]:
    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "stm32_spec"):
        return dict(base_model.stm32_spec())
    model_config = getattr(base_model, "model_config", None)
    if isinstance(model_config, dict) and model_config.get("arch"):
        return dict(model_config)
    class_name = base_model.__class__.__name__
    if class_name == "MetricGANPlusAdapter":
        return metricgan_plus_reference_arch()
    return {"arch": class_name}


def _parameter_layout(model: torch.nn.Module, arch: dict[str, Any], weight_bits: int) -> dict[str, int]:
    base_model = _unwrap_base_model(model)
    total_params = 0
    weight_params = 0
    bias_params = 0
    for parameter in base_model.parameters():
        count = int(parameter.numel())
        total_params += count
        if parameter.ndim <= 1:
            bias_params += count
        else:
            weight_params += count
    if total_params == 0 and arch.get("parameter_count_reference") is not None:
        total_params = int(arch["parameter_count_reference"])
        weight_params = int(arch.get("weight_params_reference") or max(total_params, 0))
        bias_params = int(arch.get("bias_params_reference") or max(total_params - weight_params, 0))

    weight_bytes = int(weight_params * max(weight_bits, 1) / 8)
    bias_bytes = int(bias_params * 4)
    return {
        "parameter_count": total_params,
        "weight_params": weight_params,
        "bias_params": bias_params,
        "weight_bytes": weight_bytes,
        "bias_bytes": bias_bytes,
        "flash_bytes": weight_bytes + bias_bytes,
    }


def _spectral_gating_cycles(profile: MCUProfile, erb_bands: int = 32, n_fft: int | None = None) -> int:
    fft_size = int(n_fft or profile.fft_size)
    fft_scale = _fft_cycle_scale(fft_size)
    mask_elements = (fft_size // 2 + 1) + erb_bands
    cycles = (
        int(profile.cycles_rfft512 * fft_scale)
        + int(profile.cycles_irfft512 * fft_scale)
        + profile.cycles_erb_project
        + profile.cycles_erb_expand
        + profile.cycles_window_ola
    )
    cycles += int(mask_elements * 8 * profile.cycles_eltwise_per_element)
    cycles += int(mask_elements * 2 * profile.cycles_lookup_per_element)
    return int(cycles)


def _fft_cycle_scale(fft_size: int) -> float:
    return max(fft_size, 64) / 512.0


def _frontend_workload(audio: dict[str, int], profile: MCUProfile) -> dict[str, float]:
    fft_scale = _fft_cycle_scale(audio["n_fft"])
    frontend_rfft_cycles = int(profile.cycles_rfft512 * fft_scale)
    frontend_irfft_cycles = int(profile.cycles_irfft512 * fft_scale)
    frontend_erb_project_cycles = int(profile.cycles_erb_project * (audio["erb_bands"] / profile.erb_bands))
    frontend_erb_expand_cycles = int(profile.cycles_erb_expand * (audio["erb_bands"] / profile.erb_bands))
    frontend_window_ola_cycles = int(profile.cycles_window_ola)
    frontend_dsp_cycles = (
        frontend_rfft_cycles
        + frontend_irfft_cycles
        + frontend_erb_project_cycles
        + frontend_erb_expand_cycles
        + frontend_window_ola_cycles
    )
    return {
        "frontend_rfft_cycles": frontend_rfft_cycles,
        "frontend_irfft_cycles": frontend_irfft_cycles,
        "frontend_erb_project_cycles": frontend_erb_project_cycles,
        "frontend_erb_expand_cycles": frontend_erb_expand_cycles,
        "frontend_window_ola_cycles": frontend_window_ola_cycles,
        "frontend_dsp_cycles": frontend_dsp_cycles,
    }


def _spectral_gating_workload(profile: MCUProfile, erb_bands: int = 32, n_fft: int | None = None) -> dict[str, float]:
    fft_size = int(n_fft or profile.fft_size)
    mask_elements = (fft_size // 2 + 1) + erb_bands
    eltwise_ops = int(mask_elements * 8)
    lookup_ops = int(mask_elements * 2)
    return {
        "eltwise_ops": eltwise_ops,
        "lookup_ops": lookup_ops,
        "cycles_eltwise": int(eltwise_ops * profile.cycles_eltwise_per_element),
        "cycles_lookup": int(lookup_ops * profile.cycles_lookup_per_element),
    }


def _estimate_workload(arch: dict[str, Any], profile: MCUProfile) -> dict[str, float]:
    name = str(arch.get("arch") or "")
    audio = _arch_audio_config(arch, profile)
    workload: dict[str, float] = {
        "macs_fc": 0.0,
        "macs_depthwise_conv1d": 0.0,
        "macs_pointwise_conv1d": 0.0,
        "macs_lstm": 0.0,
        "eltwise_ops": 0.0,
        "lookup_ops": 0.0,
        "cycles_fc": 0.0,
        "cycles_depthwise_conv1d": 0.0,
        "cycles_pointwise_conv1d": 0.0,
        "cycles_lstm": 0.0,
        "cycles_eltwise": 0.0,
        "cycles_lookup": 0.0,
    }
    workload.update(_frontend_workload(audio, profile))
    if name == "tiny_stm32_fc":
        dims = list(arch.get("layer_dims") or [])
        macs = sum(int(dims[i]) * int(dims[i + 1]) for i in range(max(len(dims) - 1, 0)))
        elements = sum(int(dims[i + 1]) for i in range(max(len(dims) - 1, 0)))
        workload["macs_fc"] = float(macs)
        workload["lookup_ops"] = float(elements)
        workload["cycles_fc"] = float(int(macs * profile.cycles_fc_int8_per_mac))
        workload["cycles_lookup"] = float(int(elements * profile.cycles_lookup_per_element))
    elif name == "tiny_stm32_hybrid_sg":
        dims = list(arch.get("layer_dims") or [])
        macs = sum(int(dims[i]) * int(dims[i + 1]) for i in range(max(len(dims) - 1, 0)))
        elements = sum(int(dims[i + 1]) for i in range(max(len(dims) - 1, 0)))
        gating = _spectral_gating_workload(profile, audio["erb_bands"], audio["n_fft"])
        workload["macs_fc"] = float(macs)
        workload["lookup_ops"] = float(elements + gating["lookup_ops"])
        workload["eltwise_ops"] = float(gating["eltwise_ops"])
        workload["cycles_fc"] = float(int(macs * profile.cycles_fc_int8_per_mac))
        workload["cycles_lookup"] = float(int(elements * profile.cycles_lookup_per_element) + gating["cycles_lookup"])
        workload["cycles_eltwise"] = float(gating["cycles_eltwise"])
    elif name == "tiny_stm32_tcn_hybrid":
        channels = int(arch.get("channels") or 48)
        erb_bands = int(arch.get("erb_bands") or 32)
        layers = int(arch.get("layers") or 3)
        in_channels = erb_bands * 2 + 1
        in_proj_macs = in_channels * channels
        depthwise_macs = layers * channels * 3
        pointwise_macs = layers * channels * channels
        head_macs = channels * erb_bands
        gating = _spectral_gating_workload(profile, erb_bands, audio["n_fft"])
        lookup_ops = layers * channels + erb_bands
        eltwise_ops = layers * channels * 2
        workload["macs_depthwise_conv1d"] = float(depthwise_macs)
        workload["macs_pointwise_conv1d"] = float(in_proj_macs + pointwise_macs + head_macs)
        workload["lookup_ops"] = float(lookup_ops + gating["lookup_ops"])
        workload["eltwise_ops"] = float(eltwise_ops + gating["eltwise_ops"])
        workload["cycles_depthwise_conv1d"] = float(int(depthwise_macs * profile.cycles_depthwise_conv1d_int8_per_mac))
        workload["cycles_pointwise_conv1d"] = float(
            int((in_proj_macs + pointwise_macs + head_macs) * profile.cycles_pointwise_conv1d_int8_per_mac)
        )
        workload["cycles_lookup"] = float(int(lookup_ops * profile.cycles_lookup_per_element) + gating["cycles_lookup"])
        workload["cycles_eltwise"] = float(int(eltwise_ops * profile.cycles_eltwise_per_element) + gating["cycles_eltwise"])
    elif _is_metricgan_arch(name):
        feature_bins = int(arch.get("feature_bins") or 257)
        hidden_size = int(arch.get("hidden_size") or 200)
        num_layers = int(arch.get("num_layers") or 2)
        bidirectional = bool(arch.get("bidirectional", True))
        rnn_type = str(arch.get("rnn_type") or "lstm").lower()
        gates = 3 if rnn_type == "gru" else 4
        directions = 2 if bidirectional else 1
        recurrent_macs = 0
        layer_input = feature_bins
        for _ in range(num_layers):
            recurrent_macs += directions * (gates * hidden_size * (layer_input + hidden_size + 1))
            layer_input = hidden_size * directions
        linear_dims = list(arch.get("linear_dims") or [hidden_size * directions, 300, feature_bins])
        linear_macs = sum(int(linear_dims[i]) * int(linear_dims[i + 1]) for i in range(max(len(linear_dims) - 1, 0)))
        linear_elems = sum(int(linear_dims[i + 1]) for i in range(max(len(linear_dims) - 1, 0)))
        output_elems = feature_bins * 4
        workload["macs_lstm"] = float(recurrent_macs)
        workload["macs_fc"] = float(linear_macs)
        workload["lookup_ops"] = float(linear_elems + output_elems)
        workload["eltwise_ops"] = float(output_elems)
        workload["cycles_lstm"] = float(int(recurrent_macs * profile.cycles_lstm_int8_per_mac))
        workload["cycles_fc"] = float(int(linear_macs * profile.cycles_fc_int8_per_mac))
        workload["cycles_lookup"] = float(int((linear_elems + output_elems) * profile.cycles_lookup_per_element))
        workload["cycles_eltwise"] = float(int(output_elems * profile.cycles_eltwise_per_element))
    else:
        parameter_count = int(arch.get("parameter_count_reference") or 0)
        if parameter_count <= 0:
            parameter_count = 1_000
        workload["macs_fc"] = float(parameter_count)
        workload["cycles_fc"] = float(int(parameter_count * 0.25))

    workload["macs_per_hop_total"] = float(
        workload["macs_fc"]
        + workload["macs_depthwise_conv1d"]
        + workload["macs_pointwise_conv1d"]
        + workload["macs_lstm"]
    )
    workload["cycles_per_hop"] = float(
        workload["frontend_dsp_cycles"]
        + workload["cycles_fc"]
        + workload["cycles_depthwise_conv1d"]
        + workload["cycles_pointwise_conv1d"]
        + workload["cycles_lstm"]
        + workload["cycles_eltwise"]
        + workload["cycles_lookup"]
    )
    return workload


def _estimate_sram_peak_bytes(arch: dict[str, Any], profile: MCUProfile) -> int:
    name = str(arch.get("arch") or "")
    audio = _arch_audio_config(arch, profile)
    fft_bins = audio["n_fft"] // 2 + 1
    front_end_bytes = 2 * audio["win_length"] * 2 + fft_bins * 16 + audio["erb_bands"] * 8

    if name == "tiny_stm32_fc":
        dims = list(arch.get("layer_dims") or [])
        activation_bytes = 0
        for dim in dims:
            activation_bytes += int(dim) * 2
        return int(front_end_bytes + activation_bytes + 4_096)

    if name == "tiny_stm32_hybrid_sg":
        dims = list(arch.get("layer_dims") or [])
        activation_bytes = 0
        for dim in dims:
            activation_bytes += int(dim) * 2
        guidance_bytes = audio["erb_bands"] * 2 + fft_bins * 8
        return int(front_end_bytes + activation_bytes + guidance_bytes + 6_144)

    if name == "tiny_stm32_tcn_hybrid":
        channels = int(arch.get("channels") or 48)
        layers = int(arch.get("layers") or 3)
        conv_state = channels * layers * 16
        feature_state = (audio["erb_bands"] * 2 + 1 + channels + audio["erb_bands"]) * 2
        return int(front_end_bytes + conv_state + feature_state + 8_192)

    if _is_metricgan_arch(name):
        feature_bins = int(arch.get("feature_bins") or 257)
        hidden_size = int(arch.get("hidden_size") or 200)
        num_layers = int(arch.get("num_layers") or 2)
        rnn_type = str(arch.get("rnn_type") or "lstm").lower()
        gates = 3 if rnn_type == "gru" else 4
        directions = 2 if bool(arch.get("bidirectional", True)) else 1
        sequence_frames = int(arch.get("sequence_frames") or 100)
        input_buffer = sequence_frames * feature_bins * 4
        recurrent_buffer = sequence_frames * hidden_size * directions * num_layers * 4 * 4
        gate_buffer = sequence_frames * hidden_size * directions * num_layers * gates * 2
        linear_dims = list(arch.get("linear_dims") or [hidden_size * directions, 300, feature_bins])
        linear_buffer = sequence_frames * max([feature_bins, *[int(dim) for dim in linear_dims]]) * 4
        workspace = 64 * 1024 if not bool(arch.get("non_causal", True)) else 128 * 1024
        return int(front_end_bytes + input_buffer + recurrent_buffer + gate_buffer + linear_buffer + workspace)

    return int(front_end_bytes + 8_192)


def _estimate_frequency(summary: dict[str, Any], profile: MCUProfile) -> dict[str, float]:
    hop_s = float(summary.get("hop_ms") or (profile.hop_samples / profile.sample_rate * 1000.0)) / 1000.0
    cycles_per_hop = float(summary.get("cycles_per_hop") or 0.0)
    min_required_mhz = cycles_per_hop / max(hop_s, 1e-9) / 1_000_000.0
    recommended_rt_mhz = float(max(1, math.ceil(min_required_mhz * 1.20)))
    max_profile_mhz = float(profile.cpu_hz / 1_000_000.0)
    ms_per_hop_at_recommended_mhz = cycles_per_hop / (recommended_rt_mhz * 1_000_000.0) * 1000.0
    return {
        "min_required_mhz": min_required_mhz,
        "recommended_rt_mhz": recommended_rt_mhz,
        "max_profile_mhz": max_profile_mhz,
        "ms_per_hop_at_recommended_mhz": ms_per_hop_at_recommended_mhz,
        "frequency_ok": recommended_rt_mhz <= max_profile_mhz,
    }


def _estimate_power(summary: dict[str, Any], profile: MCUProfile, frequency_summary: dict[str, float]) -> dict[str, Any]:
    hop_ms = float(summary.get("hop_ms") or (profile.hop_samples / profile.sample_rate * 1000.0))
    min_required_mhz = float(frequency_summary["min_required_mhz"])
    recommended_rt_mhz = float(frequency_summary["recommended_rt_mhz"])
    max_profile_mhz = float(frequency_summary["max_profile_mhz"])
    duty_cycle = min(max(min_required_mhz / max(recommended_rt_mhz, 1e-9), 0.0), 1.0)
    arch_name = str((summary.get("arch") or {}).get("arch") or "")
    uses_npu = bool(profile.supports_npu and arch_name not in {"spectral_gating"})
    active_engine_power_mw_max = float(profile.active_npu_power_mw if uses_npu else profile.active_cpu_power_mw)
    active_power_mw = float(
        profile.idle_power_mw
        + ((recommended_rt_mhz / max(max_profile_mhz, 1e-9)) ** 1.10) * max(active_engine_power_mw_max - profile.idle_power_mw, 0.0)
    )
    avg_power_mw = float(profile.idle_power_mw + duty_cycle * max(active_power_mw - profile.idle_power_mw, 0.0))
    energy_uj_per_hop = float(avg_power_mw * hop_ms)
    return {
        "active_engine": "npu" if uses_npu else "cpu",
        "active_engine_power_mw_max": active_engine_power_mw_max,
        "active_power_mw": active_power_mw,
        "avg_power_mw": avg_power_mw,
        "avg_power_mw_at_recommended_mhz": avg_power_mw,
        "energy_uj_per_hop": energy_uj_per_hop,
        "energy_uj_per_hop_at_recommended_mhz": energy_uj_per_hop,
        "duty_cycle_pct": duty_cycle * 100.0,
        "power_ok": avg_power_mw <= profile.power_budget_mw,
    }


def _add_summary_flags(summary: dict[str, Any], profile: MCUProfile) -> dict[str, Any]:
    hop_ms = float(summary.get("hop_ms") or (profile.hop_samples / profile.sample_rate * 1000.0))
    flash_bytes = int(summary["flash_bytes"])
    sram_peak_bytes = int(summary["sram_peak_bytes"])
    frequency_summary = _estimate_frequency(summary, profile)
    ms_per_hop_recommended = float(frequency_summary["ms_per_hop_at_recommended_mhz"])
    hardware_fit_ok = flash_bytes <= profile.flash_budget_ai and sram_peak_bytes <= profile.sram_peak_budget
    hardware_realtime_ok = ms_per_hop_recommended <= min(hop_ms, profile.compute_budget_ms)
    target_weight_ok = flash_bytes <= profile.student_weight_target_bytes
    target_weight_ideal_ok = flash_bytes <= STUDENT_WEIGHT_IDEAL_BYTES
    target_sram_ok = sram_peak_bytes <= profile.student_sram_target_bytes
    target_compute_ok = ms_per_hop_recommended <= profile.student_compute_target_ms
    power_summary = _estimate_power(summary, profile, frequency_summary)
    latency_ok = float(summary.get("lookahead_ms") or 0.0) <= PRODUCT_LOOKAHEAD_TARGET_MS
    summary.update(
        {
            "hardware_fit_ok": hardware_fit_ok,
            "hardware_realtime_ok": hardware_realtime_ok,
            "target_weight_ok": target_weight_ok,
            "target_weight_ideal_ok": target_weight_ideal_ok,
            "target_sram_ok": target_sram_ok,
            "target_compute_ok": target_compute_ok,
            "latency_ok": latency_ok,
            "fit_ok": hardware_fit_ok and target_weight_ok and target_sram_ok,
            "realtime_ok": hardware_realtime_ok,
            "cpu_load_pct": power_summary["duty_cycle_pct"],
            **frequency_summary,
            **power_summary,
        }
    )
    summary["deployment_ok"] = (
        bool(summary["fit_ok"])
        and bool(summary["frequency_ok"])
        and bool(summary["realtime_ok"])
        and bool(summary["power_ok"])
        and bool(summary["latency_ok"])
    )
    return summary


def simulate_model_fit(
    model: torch.nn.Module,
    *,
    profile_name: str = "stm32l4x6_internal_only_rt",
    weight_bits: int = 8,
    include_host_timing: bool = False,
) -> dict[str, Any]:
    profile = profile_from_name(profile_name)
    arch = _safe_model_arch(model)
    audio = _arch_audio_config(arch, profile)
    params = _parameter_layout(model, arch, weight_bits)
    workload = _estimate_workload(arch, profile)
    cycles = int(workload["cycles_per_hop"])
    sram_peak_bytes = _estimate_sram_peak_bytes(arch, profile)
    ms_per_hop_profile = cycles / profile.cpu_hz * 1000.0
    ms_per_hop_80mhz = cycles / 80_000_000 * 1000.0

    summary: dict[str, Any] = {
        "profile_name": profile.name,
        "profile_vendor": profile.vendor,
        "profile": asdict(profile),
        "arch": arch,
        "weight_bits": weight_bits,
        "parameter_count": params["parameter_count"],
        "weight_params": params["weight_params"],
        "bias_params": params["bias_params"],
        "weight_bytes": params["weight_bytes"],
        "bias_bytes": params["bias_bytes"],
        "flash_bytes": params["flash_bytes"],
        "sram_peak_bytes": sram_peak_bytes,
        "macs_per_hop_total": workload["macs_per_hop_total"],
        "macs_fc": workload["macs_fc"],
        "macs_depthwise_conv1d": workload["macs_depthwise_conv1d"],
        "macs_pointwise_conv1d": workload["macs_pointwise_conv1d"],
        "macs_lstm": workload["macs_lstm"],
        "eltwise_ops": workload["eltwise_ops"],
        "lookup_ops": workload["lookup_ops"],
        "frontend_dsp_cycles": workload["frontend_dsp_cycles"],
        "cycles_per_hop": cycles,
        "ms_per_hop_profile": ms_per_hop_profile,
        "ms_per_hop_80mhz": ms_per_hop_80mhz,
        "hop_ms": audio["hop_length"] / audio["sample_rate"] * 1000.0,
        "non_causal": bool(arch.get("non_causal", False)),
        "lookahead_ms": float(arch.get("lookahead_ms") or 0.0),
    }
    _add_summary_flags(summary, profile)

    if include_host_timing:
        device = next(iter(_unwrap_base_model(model).parameters()), None)
        device_name = device.device if device is not None else torch.device("cpu")
        length = audio["sample_rate"]
        input_tensor = torch.randn(1, 1, length, device=device_name)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        if device_name.type == "cuda":
            torch.cuda.synchronize(device_name)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        summary["host_forward_ms"] = elapsed_ms
        summary["host_realtime_factor"] = elapsed_ms / summary["hop_ms"]
    return summary


def _classic_baseline_summary(
    baseline_name: str,
    profile: MCUProfile,
    *,
    sample_rate: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    erb_bands: int | None = None,
) -> dict[str, Any]:
    normalized = baseline_name.strip().lower()
    if normalized != "spectral_gating":
        raise ValueError(f"Unsupported classic baseline for MCU simulation: {baseline_name}")
    sample_rate_value = int(sample_rate or profile.sample_rate)
    n_fft_value = int(n_fft or profile.fft_size)
    hop_length_value = int(hop_length or profile.hop_samples)
    win_length_value = int(win_length or profile.window_samples)
    erb_bands_value = int(erb_bands or profile.erb_bands)
    flash_bytes = 6 * 1024
    fft_bins = n_fft_value // 2 + 1
    sram_peak_bytes = int(2 * win_length_value * 2 + fft_bins * 16 + erb_bands_value * 8 + 16 * 1024)
    workload = _frontend_workload(
        {
            "sample_rate": sample_rate_value,
            "n_fft": n_fft_value,
            "hop_length": hop_length_value,
            "win_length": win_length_value,
            "erb_bands": erb_bands_value,
        },
        profile,
    )
    gating = _spectral_gating_workload(profile, erb_bands_value, n_fft_value)
    cycles = int(workload["frontend_dsp_cycles"] + gating["cycles_eltwise"] + gating["cycles_lookup"])
    summary = {
        "profile_name": profile.name,
        "profile_vendor": profile.vendor,
        "profile": asdict(profile),
        "baseline_name": "spectral_gating",
        "flash_bytes": flash_bytes,
        "sram_peak_bytes": sram_peak_bytes,
        "macs_per_hop_total": 0.0,
        "macs_fc": 0.0,
        "macs_depthwise_conv1d": 0.0,
        "macs_pointwise_conv1d": 0.0,
        "macs_lstm": 0.0,
        "eltwise_ops": gating["eltwise_ops"],
        "lookup_ops": gating["lookup_ops"],
        "frontend_dsp_cycles": workload["frontend_dsp_cycles"],
        "cycles_per_hop": cycles,
        "ms_per_hop_profile": cycles / profile.cpu_hz * 1000.0,
        "ms_per_hop_80mhz": cycles / 80_000_000 * 1000.0,
        "hop_ms": hop_length_value / sample_rate_value * 1000.0,
        "non_causal": False,
        "lookahead_ms": 0.0,
        "arch": {
            "arch": "spectral_gating",
            "sample_rate": sample_rate_value,
            "n_fft": n_fft_value,
            "hop_length": hop_length_value,
            "win_length": win_length_value,
            "erb_bands": erb_bands_value,
        },
    }
    _add_summary_flags(summary, profile)
    return summary


def simulate_classic_baseline(
    baseline_name: str,
    *,
    profile_name: str = "stm32l4x6_internal_only_rt",
    sample_rate: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    erb_bands: int | None = None,
) -> dict[str, Any]:
    return _classic_baseline_summary(
        baseline_name,
        profile_from_name(profile_name),
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        erb_bands=erb_bands,
    )


def simulate_metricgan_plus_reference(
    *,
    profile_name: str,
    weight_bits: int = 8,
) -> dict[str, Any]:
    profile = profile_from_name(profile_name)
    arch = metricgan_plus_reference_arch()
    params = {
        "parameter_count": int(arch["parameter_count_reference"]),
        "weight_params": int(arch["weight_params_reference"]),
        "bias_params": int(arch["bias_params_reference"]),
    }
    params["weight_bytes"] = int(params["weight_params"] * max(weight_bits, 1) / 8)
    params["bias_bytes"] = int(params["bias_params"] * 4)
    params["flash_bytes"] = params["weight_bytes"] + params["bias_bytes"]
    workload = _estimate_workload(arch, profile)
    cycles = int(workload["cycles_per_hop"])
    sram_peak_bytes = _estimate_sram_peak_bytes(arch, profile)
    summary = {
        "profile_name": profile.name,
        "profile_vendor": profile.vendor,
        "profile": asdict(profile),
        "arch": arch,
        "weight_bits": weight_bits,
        "parameter_count": params["parameter_count"],
        "weight_params": params["weight_params"],
        "bias_params": params["bias_params"],
        "weight_bytes": params["weight_bytes"],
        "bias_bytes": params["bias_bytes"],
        "flash_bytes": params["flash_bytes"],
        "sram_peak_bytes": sram_peak_bytes,
        "macs_per_hop_total": workload["macs_per_hop_total"],
        "macs_fc": workload["macs_fc"],
        "macs_depthwise_conv1d": workload["macs_depthwise_conv1d"],
        "macs_pointwise_conv1d": workload["macs_pointwise_conv1d"],
        "macs_lstm": workload["macs_lstm"],
        "eltwise_ops": workload["eltwise_ops"],
        "lookup_ops": workload["lookup_ops"],
        "frontend_dsp_cycles": workload["frontend_dsp_cycles"],
        "cycles_per_hop": cycles,
        "ms_per_hop_profile": cycles / profile.cpu_hz * 1000.0,
        "ms_per_hop_80mhz": cycles / 80_000_000 * 1000.0,
        "hop_ms": profile.hop_samples / profile.sample_rate * 1000.0,
        "non_causal": True,
        "lookahead_ms": float(arch["lookahead_ms"]),
    }
    _add_summary_flags(summary, profile)
    return summary


def _multi_profile_rollup(
    profiles: dict[str, dict[str, Any]],
    shortlist_profiles: Sequence[str],
    reference_profiles: Sequence[str],
) -> dict[str, Any]:
    supported_profiles = [
        name
        for name in shortlist_profiles
        if name in profiles
        and bool(profiles[name].get("fit_ok"))
        and bool(profiles[name].get("frequency_ok"))
        and bool(profiles[name].get("realtime_ok"))
    ]
    hardware_supported_profiles = [
        name
        for name in shortlist_profiles
        if name in profiles
        and bool(profiles[name].get("hardware_fit_ok"))
        and bool(profiles[name].get("frequency_ok"))
        and bool(profiles[name].get("hardware_realtime_ok"))
    ]
    reference_supported_profiles = [
        name
        for name in reference_profiles
        if name in profiles
        and bool(profiles[name].get("hardware_fit_ok"))
        and bool(profiles[name].get("frequency_ok"))
        and bool(profiles[name].get("hardware_realtime_ok"))
    ]
    power_supported_profiles = [
        name
        for name in shortlist_profiles
        if name in profiles
        and bool(profiles[name].get("fit_ok"))
        and bool(profiles[name].get("frequency_ok"))
        and bool(profiles[name].get("realtime_ok"))
        and bool(profiles[name].get("power_ok"))
    ]
    low_power_supported_profiles = [
        name
        for name in power_supported_profiles
        if str((profiles.get(name) or {}).get("profile", {}).get("target_class") or "") == "low_power"
    ]
    all_names = list(dict.fromkeys([*shortlist_profiles, *reference_profiles]))
    best_profile_name = None
    best_ms = None
    best_power_profile_name = None
    best_power_profile_avg_power_mw = None
    best_power_profile_recommended_rt_mhz = None
    lowest_avg_power_profile_name = None
    lowest_avg_power_mw = None
    lowest_required_mhz_profile_name = None
    lowest_required_mhz = None
    for name in all_names:
        summary = profiles.get(name)
        if summary is None:
            continue
        value = float(summary["ms_per_hop_profile"])
        if best_ms is None or value < best_ms:
            best_ms = value
            best_profile_name = name
        avg_power_mw = float(summary.get("avg_power_mw") or 0.0)
        recommended_rt_mhz = float(summary.get("recommended_rt_mhz") or 0.0)
        if lowest_avg_power_mw is None or avg_power_mw < lowest_avg_power_mw:
            lowest_avg_power_mw = avg_power_mw
            lowest_avg_power_profile_name = name
        if lowest_required_mhz is None or recommended_rt_mhz < lowest_required_mhz:
            lowest_required_mhz = recommended_rt_mhz
            lowest_required_mhz_profile_name = name
        if name in power_supported_profiles:
            if (
                best_power_profile_avg_power_mw is None
                or avg_power_mw < best_power_profile_avg_power_mw
                or (
                    avg_power_mw == best_power_profile_avg_power_mw
                    and (
                        best_power_profile_recommended_rt_mhz is None
                        or recommended_rt_mhz < best_power_profile_recommended_rt_mhz
                    )
                )
            ):
                best_power_profile_avg_power_mw = avg_power_mw
                best_power_profile_name = name
                best_power_profile_recommended_rt_mhz = recommended_rt_mhz
    return {
        "shortlist_profiles": list(shortlist_profiles),
        "reference_profiles": list(reference_profiles),
        "profiles": profiles,
        "supported_profiles": supported_profiles,
        "supported_profile_count": len(supported_profiles),
        "hardware_supported_profiles": hardware_supported_profiles,
        "hardware_supported_profile_count": len(hardware_supported_profiles),
        "reference_supported_profiles": reference_supported_profiles,
        "reference_supported_profile_count": len(reference_supported_profiles),
        "power_supported_profiles": power_supported_profiles,
        "power_supported_profile_count": len(power_supported_profiles),
        "low_power_supported_profiles": low_power_supported_profiles,
        "low_power_supported_profile_count": len(low_power_supported_profiles),
        "best_profile_name": best_profile_name,
        "best_ms_per_hop_profile": best_ms,
        "best_power_profile_name": best_power_profile_name,
        "best_power_profile_avg_power_mw": best_power_profile_avg_power_mw,
        "best_power_profile_recommended_rt_mhz": best_power_profile_recommended_rt_mhz,
        "lowest_avg_power_profile_name": lowest_avg_power_profile_name,
        "lowest_avg_power_mw": lowest_avg_power_mw,
        "lowest_required_mhz_profile_name": lowest_required_mhz_profile_name,
        "lowest_required_mhz": lowest_required_mhz,
    }


def simulate_model_across_profiles(
    model: torch.nn.Module,
    *,
    shortlist_profiles: Sequence[str] | None = None,
    reference_profiles: Sequence[str] | None = None,
    weight_bits: int = 8,
    include_host_timing: bool = False,
) -> dict[str, Any]:
    shortlist = parse_profile_names(shortlist_profiles, DEFAULT_MCU_SHORTLIST_PROFILES)
    reference = parse_profile_names(reference_profiles, DEFAULT_MCU_REFERENCE_PROFILES)
    profiles: dict[str, dict[str, Any]] = {}
    for profile_name in dict.fromkeys([*shortlist, *reference]):
        profiles[profile_name] = simulate_model_fit(
            model,
            profile_name=profile_name,
            weight_bits=weight_bits,
            include_host_timing=include_host_timing,
        )
    return _multi_profile_rollup(profiles, shortlist, reference)


def simulate_baseline_across_profiles(
    baseline_name: str,
    *,
    shortlist_profiles: Sequence[str] | None = None,
    reference_profiles: Sequence[str] | None = None,
    sample_rate: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    erb_bands: int | None = None,
) -> dict[str, Any]:
    shortlist = parse_profile_names(shortlist_profiles, DEFAULT_MCU_SHORTLIST_PROFILES)
    reference = parse_profile_names(reference_profiles, DEFAULT_MCU_REFERENCE_PROFILES)
    profiles: dict[str, dict[str, Any]] = {}
    for profile_name in dict.fromkeys([*shortlist, *reference]):
        profiles[profile_name] = simulate_classic_baseline(
            baseline_name,
            profile_name=profile_name,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            erb_bands=erb_bands,
        )
    return _multi_profile_rollup(profiles, shortlist, reference)


def simulate_metricgan_plus_reference_across_profiles(
    *,
    shortlist_profiles: Sequence[str] | None = None,
    reference_profiles: Sequence[str] | None = None,
    weight_bits: int = 8,
) -> dict[str, Any]:
    shortlist = parse_profile_names(shortlist_profiles, DEFAULT_MCU_SHORTLIST_PROFILES)
    reference = parse_profile_names(reference_profiles, DEFAULT_MCU_REFERENCE_PROFILES)
    profiles: dict[str, dict[str, Any]] = {}
    for profile_name in dict.fromkeys([*shortlist, *reference]):
        profiles[profile_name] = simulate_metricgan_plus_reference(
            profile_name=profile_name,
            weight_bits=weight_bits,
        )
    return _multi_profile_rollup(profiles, shortlist, reference)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate MCU resource fit and real-time behavior.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-family", default="tiny_stm32_fc")
    parser.add_argument("--variant", default="small")
    parser.add_argument("--profile", default="stm32l4x6_internal_only_rt")
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--erb-bands", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--guidance-classic", default="none")
    parser.add_argument("--qat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--include-host-timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.profiles:
        if args.model_family == "metricgan_plus":
            result = simulate_metricgan_plus_reference_across_profiles(
                shortlist_profiles=parse_profile_names(args.profiles, DEFAULT_MCU_SHORTLIST_PROFILES),
                reference_profiles=DEFAULT_MCU_REFERENCE_PROFILES,
                weight_bits=args.weight_bits,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
    if args.checkpoint:
        from sebench.checkpoints import load_model_from_checkpoint

        model, _ = load_model_from_checkpoint(args.checkpoint, device=args.device)
    else:
        from sebench.models import build_enhancer

        model = build_enhancer(
            args.model_family,
            args.variant,
            erb_bands=args.erb_bands,
            context_frames=args.context_frames,
            guidance_classic=args.guidance_classic,
            qat=args.qat,
        )
        model.to(args.device)

    if args.profiles:
        result = simulate_model_across_profiles(
            model,
            shortlist_profiles=parse_profile_names(args.profiles, DEFAULT_MCU_SHORTLIST_PROFILES),
            reference_profiles=DEFAULT_MCU_REFERENCE_PROFILES,
            weight_bits=args.weight_bits,
            include_host_timing=args.include_host_timing,
        )
    elif args.model_family == "metricgan_plus" and args.checkpoint is None:
        result = simulate_metricgan_plus_reference(
            profile_name=args.profile,
            weight_bits=args.weight_bits,
        )
    else:
        result = simulate_model_fit(
            model,
            profile_name=args.profile,
            weight_bits=args.weight_bits,
            include_host_timing=args.include_host_timing,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
