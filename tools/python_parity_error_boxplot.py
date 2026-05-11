#!/usr/bin/env python3
"""Generate a Rust-vs-python-itu-r numeric parity box plot.

This is a manual documentation tool, not a test runner. It expects the local
`itu_rs` Python extension to be installed in the active environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "images" / "python-parity-error-boxplot.png"


@dataclasses.dataclass(frozen=True)
class Sample:
    index: int
    lat: float
    lon: float
    freq: float
    elevation: float
    p: float
    dish_m: float
    tau: float
    hs_km: float
    pressure_hpa: float
    temp_k: float
    rho_gm3: float
    v_t_kgm2: float
    rain_rate_mmh: float


@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    unit: str
    compare: Callable[[Sample], tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the README box plot for itu-rs numeric parity against python-itu-r."
    )
    parser.add_argument("--samples", type=int, default=96, help="deterministic fuzz samples to run")
    parser.add_argument("--seed", type=int, default=20260511, help="deterministic RNG seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="comparison worker threads; use 1 for serial execution",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path",
    )
    parser.add_argument(
        "--itu-rpy-path",
        type=Path,
        default=Path(os.environ["ITU_RPY_PATH"]) if "ITU_RPY_PATH" in os.environ else None,
        help="checkout path for the reference python-itu-r package",
    )
    return parser.parse_args()


def configure_python_itu_r(path: Path | None) -> None:
    if path is not None and path.exists():
        sys.path.insert(0, str(path.resolve()))


def scalar(value: object) -> float:
    import numpy as np

    if hasattr(value, "value"):
        value = value.value
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def require_modules() -> tuple[object, object, object, object, object, object, object, object, object]:
    try:
        import itu_rs
    except ImportError as exc:
        raise SystemExit(
            "Could not import `itu_rs`. Build the local Python extension first:\n"
            "  cd python && uv run --group dev maturin develop --manifest-path Cargo.toml"
        ) from exc

    try:
        import itur
        from itur.models import itu1510, itu618, itu676, itu837, itu838, itu839, itu840
    except ImportError as exc:
        raise SystemExit(
            "Could not import the reference `python-itu-r` package. Set ITU_RPY_PATH "
            "to its checkout, or install it in the active Python environment."
        ) from exc

    return itu_rs, itur, itu1510, itu618, itu676, itu837, itu838, itu839, itu840


def make_samples(count: int, seed: int) -> list[Sample]:
    if count < 2:
        raise SystemExit("--samples must be at least 2")

    rng = random.Random(seed)
    p_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    tau_values = [0.0, 45.0, 90.0]
    samples = []
    for index in range(count):
        samples.append(
            Sample(
                index=index,
                lat=rng.uniform(-55.0, 55.0),
                lon=rng.uniform(-175.0, 175.0),
                freq=rng.uniform(6.0, 40.0),
                elevation=rng.uniform(8.0, 75.0),
                p=rng.choice(p_values),
                dish_m=rng.uniform(0.6, 3.0),
                tau=rng.choice(tau_values),
                hs_km=rng.uniform(0.01, 1.2),
                pressure_hpa=rng.uniform(930.0, 1030.0),
                temp_k=rng.uniform(270.0, 305.0),
                rho_gm3=rng.uniform(2.0, 20.0),
                v_t_kgm2=rng.uniform(5.0, 55.0),
                rain_rate_mmh=rng.uniform(0.1, 80.0),
            )
        )
    return samples


def build_cases() -> list[Case]:
    itu_rs, itur, itu1510, itu618, itu676, itu837, itu838, itu839, itu840 = require_modules()

    return [
        Case(
            "P.1510 surface temp",
            "K",
            lambda s: (
                itu_rs.surface_mean_temperature_k(s.lat, s.lon),
                scalar(itu1510.surface_mean_temperature(s.lat, s.lon)),
            ),
        ),
        Case(
            "P.837 rainfall rate",
            "mm/h",
            lambda s: (
                itu_rs.rainfall_rate_mmh(s.lat, s.lon, s.p),
                scalar(itu837.rainfall_rate(s.lat, s.lon, s.p)),
            ),
        ),
        Case(
            "P.839 rain height",
            "km",
            lambda s: (
                itu_rs.rain_height_km(s.lat, s.lon),
                scalar(itu839.rain_height(s.lat, s.lon)),
            ),
        ),
        Case(
            "P.838 rain spec. att.",
            "dB/km",
            lambda s: (
                itu_rs.rain_specific_attenuation_db_per_km(
                    s.rain_rate_mmh, s.freq, s.elevation, s.tau
                ),
                scalar(
                    itu838.rain_specific_attenuation(
                        s.rain_rate_mmh, s.freq, s.elevation, s.tau
                    )
                ),
            ),
        ),
        Case(
            "P.840 cloud att.",
            "dB",
            lambda s: (
                itu_rs.cloud_attenuation_db(s.lat, s.lon, s.elevation, s.freq, s.p),
                scalar(itu840.cloud_attenuation(s.lat, s.lon, s.elevation, s.freq, s.p)),
            ),
        ),
        Case(
            "P.676 gamma exact",
            "dB/km",
            lambda s: (
                itu_rs.gamma_exact_db_per_km(s.freq, s.pressure_hpa, s.rho_gm3, s.temp_k),
                scalar(itu676.gamma_exact(s.freq, s.pressure_hpa, s.rho_gm3, s.temp_k)),
            ),
        ),
        Case(
            "P.676 gas slant",
            "dB",
            lambda s: (
                itu_rs.gaseous_attenuation_slant_path_db(
                    s.freq,
                    s.elevation,
                    s.rho_gm3,
                    s.pressure_hpa,
                    s.temp_k,
                    s.v_t_kgm2,
                    s.hs_km,
                    False,
                ),
                scalar(
                    itu676.gaseous_attenuation_slant_path(
                        s.freq,
                        s.elevation,
                        s.rho_gm3,
                        s.pressure_hpa,
                        s.temp_k,
                        s.v_t_kgm2,
                        s.hs_km,
                        mode="approx",
                    )
                ),
            ),
        ),
        Case(
            "P.618 rain att.",
            "dB",
            lambda s: (
                itu_rs.rain_attenuation_db(
                    s.lat, s.lon, s.freq, s.elevation, s.hs_km, s.p, None, s.tau, None
                ),
                scalar(
                    itu618.rain_attenuation(
                        s.lat, s.lon, s.freq, s.elevation, s.hs_km, s.p, None, s.tau, None
                    )
                ),
            ),
        ),
        Case(
            "P.618 scintillation",
            "dB",
            lambda s: (
                itu_rs.scintillation_attenuation_db(
                    s.lat,
                    s.lon,
                    s.freq,
                    s.elevation,
                    s.p,
                    s.dish_m,
                    0.5,
                    None,
                    None,
                    None,
                    1000.0,
                ),
                scalar(
                    itu618.scintillation_attenuation(
                        s.lat,
                        s.lon,
                        s.freq,
                        s.elevation,
                        s.p,
                        s.dish_m,
                        0.5,
                        None,
                        None,
                        None,
                        1000.0,
                    )
                ),
            ),
        ),
        Case(
            "Full slant path",
            "dB",
            lambda s: (
                itu_rs.atmospheric_attenuation_slant_path(
                    s.lat, s.lon, s.freq, s.elevation, s.p, s.dish_m
                ).total_db,
                scalar(
                    itur.atmospheric_attenuation_slant_path(
                        s.lat, s.lon, s.freq, s.elevation, s.p, s.dish_m
                    )
                ),
            ),
        ),
    ]


def compare_sample(sample: Sample, cases: list[Case]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for case in cases:
        rust_value, python_value = case.compare(sample)
        if not math.isfinite(rust_value) or not math.isfinite(python_value):
            raise ValueError(
                f"{case.label} produced a non-finite value for sample {sample.index}: "
                f"rust={rust_value!r}, python={python_value!r}"
            )
        errors[case.label] = abs(rust_value - python_value)
    return errors


def collect_errors(samples: list[Sample], cases: list[Case], workers: int) -> dict[str, list[float]]:
    errors = {case.label: [] for case in cases}

    first, rest = samples[0], samples[1:]
    for label, error in compare_sample(first, cases).items():
        errors[label].append(error)

    if workers <= 1:
        for sample in rest:
            for label, error in compare_sample(sample, cases).items():
                errors[label].append(error)
        return errors

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compare_sample, sample, cases): sample.index for sample in rest}
        for future in concurrent.futures.as_completed(futures):
            sample_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"comparison failed for sample {sample_index}") from exc
            for label, error in result.items():
                errors[label].append(error)

    return errors


def stats(values: Iterable[float]) -> tuple[float, float, float]:
    items = list(values)
    mean = sum(items) / len(items)
    variance = sum((item - mean) ** 2 for item in items) / len(items)
    return mean, math.sqrt(variance), max(items)


def render_plot(errors: dict[str, list[float]], cases: list[Case], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [case.label for case in cases]
    units = {case.label: case.unit for case in cases}
    data = [[max(error, 1e-16) for error in errors[label]] for label in labels]
    sigma = {label: stats(errors[label])[1] for label in labels}
    tick_labels = [f"{label}\nstd={sigma[label]:.1e} {units[label]}" for label in labels]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    try:
        plot = ax.boxplot(data, tick_labels=tick_labels, showfliers=True, patch_artist=True)
    except TypeError:
        plot = ax.boxplot(data, labels=tick_labels, showfliers=True, patch_artist=True)

    for box in plot["boxes"]:
        box.set_facecolor("#9ecae1")
        box.set_edgecolor("#08519c")
        box.set_alpha(0.85)

    ax.set_yscale("log")
    ax.set_ylabel("absolute error vs python-itu-r (native units, log scale)")
    ax.set_title("itu-rs numerical parity across deterministic fuzz samples")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def print_summary(errors: dict[str, list[float]], cases: list[Case], workers: int, output: Path) -> None:
    print(f"workers: {workers}")
    print(f"output: {output}")
    print()
    print("| Case | n | mean abs error | std dev | max abs error |")
    print("|---|---:|---:|---:|---:|")
    for case in cases:
        mean, sigma, maximum = stats(errors[case.label])
        unit = case.unit
        print(
            f"| {case.label} | {len(errors[case.label])} | "
            f"{mean:.6e} {unit} | {sigma:.6e} {unit} | {maximum:.6e} {unit} |"
        )


def main() -> None:
    args = parse_args()
    configure_python_itu_r(args.itu_rpy_path)
    cases = build_cases()
    samples = make_samples(args.samples, args.seed)
    workers = max(1, args.workers)

    errors = collect_errors(samples, cases, workers)
    render_plot(errors, cases, args.output)
    print_summary(errors, cases, workers, args.output)


if __name__ == "__main__":
    main()
