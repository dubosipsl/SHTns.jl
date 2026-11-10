# CUDA single precision

Branch: `codex/cuda-float32`, based on upstream `db5168a`.

The CUDA extension supports Float32 spatial buffers and ComplexF32 spectral
buffers for scalar, tangential-vector and radial/tangential-vector transforms,
including batched and allocating/in-place calls. Set `float32=true` on the
configuration. Float64 remains the default. A mismatched CUDA plan precision is
rejected before calling C. The C `ltr` argument is declared as `Cint`, matching
`shtns_cuda.h`.

## GPU tests

`test/ext/float32.jl` compares the public CUDA API against independent CPU
Float64 transforms, for both precisions, batches of 1 and 3, supported layouts,
padding, flat/matrix spectra and all three transform variants. It also checks
input preservation by allocating analysis and precision mismatch rejection.
It is included by `test/ext/cudaext.jl`, and can be run separately in a Julia
environment containing SHTns and CUDA.

## ClimFlowsBenchmarks

The companion branch `codex/shtns-cuda-float32` contains a standalone prototype,
a development environment, a verification script and recorded results.
The benchmark uses SHTns_jll 3.7.0+0, Julia 1.11.9, CUDA runtime 12.9, and the
NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU (compute capability 12.0).

SHTns' default Float32 analysis uses atomic additions across latitude blocks.
At T255, repeating the direct velocity-to-divergence/vorticity pipeline differed
by about 3e-5 in relative L2, exceeding the benchmark's 1e-5 graph/direct check.
The benchmark adapter uses SHTns' existing `sht_gpu.conf` tuning mechanism for
that case only: `[nwarp_s,nf_s,nw_s,nwarp_a,nf_a,lspan_a]=[4,4,2,6,4,12]`.
The 192-thread analysis block covers all 192 northern latitudes and eliminates
inter-block atomic accumulation. This tuning is not part of the Julia wrapper,
and neither the analytic tolerances nor the graph/direct threshold is relaxed.
The benchmark records the profile in each result's notes.
