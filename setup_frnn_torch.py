#!/usr/bin/env python3
# setup_frnn_torch.py — build the zero-copy PyTorch CUDA extension `frnn_torch`.
#
#   module load pytorch/2.6.0
#   python3 setup_frnn_torch.py build_ext --inplace
#
# Produces frnn_torch*.so in the repo root, importable as `import frnn_torch`.
# Compiles only the active N-D engine kernels (no xju2 / external prefix-sum: the
# engine uses thrust::exclusive_scan, so the custom scan is not needed here).
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="frnn_torch",
    ext_modules=[
        CUDAExtension(
            name="frnn_torch",
            sources=[
                "python_interface/frnn_torch.cu",
                "frnn/csrc/grid/insert_points.cu",
                "frnn/csrc/grid/find_nbrs.cu",
                "frnn/csrc/no_grid_frnn/no_grid_frnn.cu",
                "frnn/csrc/projection/verify.cu",
                "frnn/csrc/projection/project.cu",
            ],
            include_dirs=[
                ROOT,
                os.path.join(ROOT, "frnn/csrc/grid"),
                os.path.join(ROOT, "frnn/csrc/no_grid_frnn"),
                os.path.join(ROOT, "frnn/csrc/utils"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                # A100 (Perlmutter). --expt-relaxed-constexpr keeps thrust happy.
                "nvcc": ["-O3", "-arch=sm_80", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
