#!/usr/bin/env python3
"""
discover_frnn_opts.py — Drive SkyDiscover's Adaptive Evolution (AdaEvolve) to find
LOW-LEVEL CUDA optimization points in frnn/csrc/grid/find_nbrs.cu.
"""

import argparse
import os
import re
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

You are optimizing frnn/csrc/grid/find_nbrs.cu — the fixed-radius nearest-neighbor search kernel for an N-dimensional point cloud engine running on NVIDIA A100 (sm_80). This is the hottest code path: for every query point it traverses neighboring grid cells and maintains a max-heap of the K closest points found so far.

Focus your mutations on these high-impact areas:
1. **Shared memory tiling**: load candidate points into shared memory per cell to reduce redundant global reads across threads searching the same cell.
2. **__ldg() read-only cache**: ensure all read-only global loads (points array, cell offsets, cell counts) go through the read-only cache for warp broadcast.
3. **Warp divergence**: the per-dimension neighbor cell loop and the heap insert_neighbor() call are divergence sources — restructure to minimize thread divergence within a warp.
4. **Register pressure**: local_dists[K] and local_idxs[K] are sized to MAX_K_CAPACITY=128 even when K=16 — consider templating on K to let the compiler size them correctly.
5. **Loop unrolling**: the innermost distance accumulation loop over dimensions (D=3 for the grid path) can be fully unrolled with #pragma unroll.
6. **Memory coalescing**: the SoA layout p[d*P + i] is already coalesced — preserve it in all mutations.

All code must be valid CUDA C++ compatible with sm_80. Do not introduce any Python, PyTorch, or high-level library dependencies. Mutations must preserve the exact function signatures and semantics of FindNbrsKernel and UnpermuteSortedOutput.

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
        bench_cmd = "module load pytorch/2.8.0 && D_SWEEP=3 PYTHONPATH=. python3 Tests/benchmark_master.py"
        res = subprocess.run(bench_cmd, shell=True, capture_output=True, text=True, cwd=str(_REPO))
        if res.returncode != 0:
            return {"combined_score": 0.0, "artifacts": {"feedback": f"Benchmark Run Crashed:\n{res.stderr}"}}

        # 4. Extract FRNN latencies from lines like:
        #    FRNN=0.20ms  FAISS=1.6ms  PyG=2.4ms  xju2=0.9ms
        frnn_times = re.findall(r'FRNN=([\d.]+)ms', res.stdout)
        if frnn_times:
            latency = sum(float(t) for t in frnn_times) / len(frnn_times)
        else:
            latency = 100.0  # Fallback

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

    print(f"[discover] Kicking off Low-Level CUDA Sweep on find_nbrs optimization...")
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