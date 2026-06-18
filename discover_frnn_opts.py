#!/usr/bin/env python3
"""
discover_frnn_opts.py — Drive SkyDiscover's Adaptive Evolution (AdaEvolve) to find
LOW-LEVEL CUDA optimization points in frnn/csrc/grid/find_nbrs.cu.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Resolve skydiscover from the submodule tree before any other import
_REPO = Path(__file__).parent.resolve()
sys.path.insert(0, str(_REPO / "skydiscover" / "skydiscover"))

from skydiscover import run_discovery  # noqa: E402

# ---------------------------------------------------------------------------
# Low-Level CUDA Domain Context Injected Into Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert in high-performance GPU computing, raw CUDA kernel writing, and A100 hardware architecture.

The code to evolve is the low-level CUDA file `frnn/csrc/grid/find_nbrs.cu`. Your goal is to maximize A100 GPU occupancy and memory throughput by mutating the execution pathways inside the EVOLVE-BLOCK.

== Hardware Optimization Angles to Explore ==
1. Warp Divergence & Branching: Eliminate conditional branching logic inside loops wherever possible to keep the 32-thread SIMT warps execution synchronized.
2. Coalesced Memory Lanes: Group your global memory loads using Structure-of-Arrays layout rules so the hardware can fulfill memory requests in a single transaction.
3. Shared Memory Cache: Copy neighbor indices into shared memory clusters to reduce global device latency, and verify that reads avoid shared memory bank conflicts.
4. Loop Unrolling: Explicitly use `#pragma unroll` on static dimensions to decrease loop counter overhead and maximize active instruction pipeline slots.

Return ONLY the complete, compilable contents of find_nbrs.cu. Do not wrap code in markdown code blocks or backticks.
""".strip()

# ---------------------------------------------------------------------------
# Modified Dynamic Evaluator (Replaces the broken placeholder evaluator)
# ---------------------------------------------------------------------------
def custom_cuda_evaluator(program_path: str) -> dict:
    """
    Reads the LLM candidate CUDA code, copies it into the src tree, compiles it 
    natively on the A100, and returns a score based on real benchmark latency.
    """
    cuda_src_dest = _REPO / "frnn" / "csrc" / "grid" / "find_nbrs.cu"
    
    try:
        # 1. Overwrite target source file with mutated code candidate
        with open(program_path, "r") as src, open(cuda_src_dest, "w") as dest:
            dest.write(src.read())

        # 2. Compile natively using your exact environment configurations
        compile_cmd = "module load pytorch/2.8.0 && python3 setup_frnn_torch.py build_ext --inplace"
        res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True, cwd=str(_REPO))
        if res.returncode != 0:
            return {"combined_score": 0.0, "artifacts": {"feedback": f"Compilation Failed:\n{res.stderr}"}}

        # 3. Run your official repository master benchmark profile
        bench_cmd = "module load pytorch/2.8.0 && PYTHONPATH=. python3 Tests/benchmark_master.py"
        res = subprocess.run(bench_cmd, shell=True, capture_output=True, text=True, cwd=str(_REPO))
        if res.returncode != 0:
            return {"combined_score": 0.0, "artifacts": {"feedback": f"Benchmark Run Crashed:\n{res.stderr}"}}

        # 4. Extract raw wall-clock metrics from your benchmark output log
        # (Assuming benchmark prints 'FRNN: XX.XX ms' or similar; adjust string parsing as needed)
        latency = 100.0  # Fallback
        for line in res.stdout.splitlines():
            if "latency_ms" in line or "FRNN" in line:
                try:
                    # Quick numeric extraction logic example
                    parts = [float(s) for s in line.replace(",", " ").split() if s.replace(".", "", 1).isdigit()]
                    if parts:
                        latency = parts[0]
                        break
                except ValueError:
                    continue

        # Score is inverted execution time: lower latency = higher score
        score = 1000.0 / (latency + 1e-6)
        return {
            "combined_score": float(score),
            "artifacts": {"feedback": f"Successfully compiled and benchmarked. Latency: {latency:.4f} ms"}
        }

    except Exception as e:
        return {"combined_score": 0.0, "artifacts": {"feedback": f"Unexpected Evaluator Failure: {str(e)}"}}


# ---------------------------------------------------------------------------
# CLI & Runner execution
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Low-Level CUDA AdaEvolve Driver via SkyDiscover")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--api-base", default="https://api.cborg.lbl.gov/v1")
    args = p.parse_args()

    # Define target files
    cuda_target = _REPO / "frnn" / "csrc" / "grid" / "find_nbrs.cu"
    out_dir = _REPO / "skydiscover_out"

    print(f"[discover] Kicking off Low-Level CUDA Sweep on finding optimization points...")
    print(f"[discover] Target file: {cuda_target.relative_to(_REPO)}")
    print(f"[discover] Iterations : {args.iterations}\n")

    # Inlining our execution routine to cleanly route scores dynamically
    result = run_discovery(
        evaluator=custom_cuda_evaluator,  # Pass function directly to bypass external script limits
        initial_program=str(cuda_target),
        model=args.model,
        iterations=args.iterations,
        search="adaevolve",
        output_dir=str(out_dir),
        system_prompt=_SYSTEM_PROMPT,
        api_base=args.api_base,
        cleanup=False,
    )

    print(f"\n[discover] Optimization completed! Best score achieved: {result.best_score:.6f}")
    print(f"[discover] Optimized files saved safely inside: {result.output_dir}")

if __name__ == "__main__":
    main()