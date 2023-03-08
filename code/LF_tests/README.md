# README

This document provides a high level overview of the contents of the directory.

## Warnings and General Notes

Unless otherwise stated, the `SBATCH` commands that direct output will need to be modified, since we cannot use macros here. We created a Python environment prior to building the code because certain versions of Numba's dependencies cause the builds to fail. Check the compilers in the build scripts to ensure they match your system.

## Scipts to Build PyMFEM and Launch Slurm Jobs

* `slurm_pymfem_build_cpu.sb`: Builds PyMFEM with MPI support.

* `slurm_pymfem_build_gpu.sb`: Builds PyMFEM with CUDA support.

* `slurm_parla_cpu_runs.sb`: Job script that launches a strong scaling test for the Parla-CPU example code. The problem size is hard coded to the file, but this can easily be modified.

* `slurm_parla_gpu_runs.sb`: Job script that launches a strong scaling test for the Parla-GPU example code. This code calls the multi-GPU implementation. The problem size is hard coded to the file, but this can easily be modified. To verify correctness and establish a baseline for performance, we compare with the fast assembly method from PyMFEM using the CUDA build.

* `slurm_pymfem_cpu_runs.sb`: Job script that launches a CPU strong scaling test for the pure PyMFEM example. This test uses MPI for parallelism. No Parla code is called here.

## Unit Tests

* `numba_gpu_test.py`: Simple test to check whether or not the CUDA environment is properly setup. This particular test launches a kernel that increments the entries of a 2-D array.

* `PArray_coherence_test.py`: Tests the coherence of PArray in the context of a multi-device reduction using Parla. Specifically, we show that unless the PArray is cleared or reinitialized between trials, then its data becomes contaminated from previous trials.

* `PArray_mgpu_reduction_v1.py`: Tests the evaluation of a multi-device reduction using PArray. The reduction involves a multi-dimensional array and sums across the device memory, storing the final result in a CuPy array. Coherence is enforced by recreating the PArray between trials. 

* `PArray_mgpu_reduction_v2.py`: Same thing as version 1, but the result is stored in a PArray instead of a CuPy array.


## Linear Form Tests

* `derived_LFIntegrator_ex1.py`: Script that shows how to create new integrators for the abstract LinearForm class from MFEM. No Parla is used here, and this approach is not useful because we lose many opportunities for parallelism here.

* `parla_LFIntegrator_ex1_naive.py`: This is the simplest example that combines MFEM and Parla together. This example is also considerably slower than others because the Parla tasks take the hit from the interpreter.

* `parla_LFIntegrator_ex1_cpu.py`: Similar to the previous example, except that the Parla tasks make calls to kernels "jitted" with Numba instead of making interpreted calls. PArray is not used here, but this is something that could be explored.

* `parla_LFIntegrator_ex1_sgpu.py`: First pass at trying to combine CUDA with Parla and PyMFEM. This example uses CuPy arrays and is written with only a single device in mind. We don't really use this anymore, but it was useful as a baseline for performance.

* `parla_LFIntegrator_ex1_mgpu.py`: Extension of the previous code to handle multiple GPU devices. Rather than manually transfer between devices, we opted for automatic transfers with PArray. This use case largely motivated the unit tests in the previous section. This is where we can make the most progress. The PArrays have not been reset in this code, so that needs to be done first, to ensure correctness.