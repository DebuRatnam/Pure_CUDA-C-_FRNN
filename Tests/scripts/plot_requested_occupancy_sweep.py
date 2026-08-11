#!/usr/bin/env python3
"""Merge the requested new cells with previously measured D=12/D=16 cells."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


EXISTING = {
    "D12_N200000": {"latency_ms": 7.3781710234470665, "libfrnn_gpu_ms": 5.978823988698423},
    "D12_N500000": {"latency_ms": 23.22301553795114, "libfrnn_gpu_ms": 21.653761039488018},
    "D12_N1000000": {"latency_ms": 59.32024650974199, "libfrnn_gpu_ms": 68.0998710449785},
    "D16_N200000": {"latency_ms": 20.085041061975062, "libfrnn_gpu_ms": 9.88839496858418},
    "D16_N500000": {"latency_ms": 43.707173492293805, "libfrnn_gpu_ms": 28.637330513447523},
    "D16_N1000000": {"latency_ms": 98.8155270460993, "libfrnn_gpu_ms": 90.10353952180594},
}


def main():
    test_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="+",
        default=[str(test_root / "json_results" / "requested_new_cells.partial.json")],
    )
    parser.add_argument(
        "--output",
        default=str(test_root / "png_results" / "requested_occupancy_sweep.png"),
    )
    args = parser.parse_args()

    results = dict(EXISTING)
    for path in args.results:
        with open(path) as f:
            results.update(json.load(f))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    for ax, dim in zip(axes[0], (3, 12, 16)):
        cells = sorted(
            (int(key.split("_N")[1]), value)
            for key, value in results.items()
            if key.startswith(f"D{dim}_N")
        )
        for label, field, color, marker in (
            ("FRNN", "latency_ms", "#1f77b4", "o"),
            ("libFRNN", "libfrnn_gpu_ms", "#d62728", "^"),
        ):
            points = [(n / 1e6, value.get(field)) for n, value in cells
                      if value.get(field) is not None]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points],
                        marker=marker, linewidth=2, label=label, color=color)
        ax.set_title(f"D = {dim}")
        ax.set_xlabel("Points (millions)")
        ax.set_ylabel("Median latency (ms)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    fig.suptitle("High-dimensional occupancy sweep — low-rank, K=16")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
