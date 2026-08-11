# High-dimensional grid occupancy tuning

The D=12, D=16, and generic high-dimensional kernels have independent CMake
launch policies. Supported block sizes are 64, 128, and 256 threads. Do not
select a winner from theoretical occupancy alone: rebuild, validate, and time
every candidate on the target GPU.

## Build a policy and capture ptxas output

For example, this builds D=12 at 128 threads / 8 minimum blocks per SM while
leaving the other policies explicit:

```bash
CMAKE_ARGS="\
  -DFRNN_GRID_D12_THREADS=128 \
  -DFRNN_GRID_D12_MIN_BLOCKS=8 \
  -DFRNN_GRID_D16_THREADS=128 \
  -DFRNN_GRID_D16_MIN_BLOCKS=8 \
  -DFRNN_GRID_GENERIC_THREADS=128 \
  -DFRNN_GRID_GENERIC_MIN_BLOCKS=4 \
  -DFRNN_PTXAS_VERBOSE=ON" \
pip install --user --no-build-isolation -e . 2>&1 | tee ptxas-128-8.log
```

Inspect every `FindNbrsGridDimKernel` entry, not only the aggregate compiler
output. The mangled template arguments contain `CAP`, grid dimension, full
dimension, threads, and minimum blocks in that order. CAP=16 is the benchmark's
K=16 path.

The initial sm_80 compile-only sweep produced these CAP=16/grid-D=4 results:

| Threads | Min blocks | D=12 registers | D=12 spills (store/load bytes) | D=16 registers | D=16 spills (store/load bytes) |
|---:|---:|---:|---:|---:|---:|
| 64 | 4, 8, or 16 | 56 | 0 / 0 | 64 | 0 / 0 |
| 128 | 4 or 8 | 56 | 0 / 0 | 64 | 0 / 0 |
| 128 | 16 | 32 | 72 / 64 | 32 | 124 / 136 |
| 256 | 2 or 4 | 56 | 0 / 0 | 64 | 0 / 0 |
| 256 | 8 | 32 | 72 / 64 | 32 | 124 / 136 |

This only eliminates obviously spill-prone candidates. It is not a runtime
ranking.

## Theoretical occupancy and benchmark

The CUDA occupancy API report is opt-in and prints once for each selected
kernel specialization:

```bash
FRNN_REPORT_OCCUPANCY=1 \
EXTRA_CELLS=12:200000 N_SWEEP= D_SWEEP= \
PYTHONPATH=. python3 Tests/benchmark_vs_libfrnn.py 2>&1 | tee occupancy-128-8.log
```

The report includes CAP, dimensions, threads, minimum blocks, registers, local
memory, active blocks per SM, and theoretical occupancy. Run correctness tests
after every rebuild before accepting its latency.

## Achieved occupancy with Nsight Compute

Profile one validated policy at a time. The kernel-name filter excludes setup
and libFRNN kernels:

```bash
ncu --kernel-name 'regex:FindNbrsGridDimKernel.*' \
  --metrics launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active \
  --target-processes all \
  python3 Tests/benchmark_vs_libfrnn.py 2>&1 | tee ncu-128-8.log
```

Start runtime comparisons with the strongest spill-free constraints from the
compile sweep: 64/16, 128/8, and 256/4. Also measure weaker minimum-block values
if ptxas changes code generation. Use identical data, radius, warmup, and trial
counts, and retain separate D=12, D=16, and generic policies when the results
justify doing so.

Commit only the validated measured winner, independently, with the subject:

```text
Tune occupancy for high-dimensional grid search
```
