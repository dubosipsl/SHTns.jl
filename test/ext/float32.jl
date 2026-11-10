# Run explicitly in a CUDA-enabled environment containing SHTns and CUDA.
using Test, Random, LinearAlgebra, CUDA, SHTns

CUDA.allowscalar(false)

# Initialize the on-device API on Julia's stream, as in ClimFlowsBenchmarks.
function device_config(T, B, contiguous_phi, padding)
    cfg = SHTnsCfg(21, 21, 1, 32, 64;
        howmany=B, shtype=SHTns.QuickInit(; float32=T === Float32, contiguous_phi, padding))
    stream = CUDA.stream()
    ccall((:cushtns_set_streams, SHTns.libshtns), Cvoid,
        (SHTns.shtns_cfg, CUDA.CUstream, CUDA.CUstream), cfg.cfg, stream, C_NULL)
    device = ccall((:cushtns_init_gpu, SHTns.libshtns), Cint, (SHTns.shtns_cfg,), cfg.cfg)
    device >= 0 || error("Cannot initialize SHTns CUDA")
    cfg.shtype = SHTns.QuickInit(; gpu=true, float32=T === Float32, contiguous_phi, padding)
    return cfg
end

function physical(cfg, x)
    a = Array(x)
    return cfg.shtype.contiguous_phi ? a[:, 1:cfg.nlat, :] : a[1:cfg.nlat, :, :]
end
relative(a, b) = norm(a - b) / norm(b)

@testset "CUDA Float32 and Float64 public API" begin
    rng = MersenneTwister(20261110)
    for T in (Float32, Float64), B in (1, 3), phi in (false, true), padding in (false, true)
        B > 1 && phi && continue
        @testset "$T batch=$B contiguous_phi=$phi padding=$padding" begin
            cfg = device_config(T, B, phi, padding)
            cpu = SHTnsCfg(21, 21, 1, 32, 64; howmany=B,
                shtype=SHTns.QuickInit(; contiguous_phi=phi, padding))
            q = randn(rng, ComplexF64, nlm(cfg), B)
            q[cfg.mi .== 0, :] .= real.(q[cfg.mi .== 0, :])
            s = copy(q); s[1, :] .= 0
            t = 0.7 .* s
            tol = T === Float32 ? 5e-4 : 5e-10
            for flat in (false, true)
                host = map(x -> flat || B == 1 ? vec(x) : x, (q, s, t))
                gpu = map(x -> CuArray(Complex{T}.(x)), host)
                for components in (1, 2, 3)
                    h = components == 1 ? (host[1],) : components == 2 ? (host[2], host[3]) : host
                    g = components == 1 ? (gpu[1],) : components == 2 ? (gpu[2], gpu[3]) : gpu
                    reference = SHTns.synth(cpu, h...)
                    reference = components == 1 ? (reference,) : reference
                    spatial = SHTns.synth(cfg, g...)
                    spatial = components == 1 ? (spatial,) : spatial
                    CUDA.device_synchronize()
                    @test all(eltype(x) === T for x in spatial)
                    for (actual, expected) in zip(spatial, reference)
                        @test relative(physical(cfg, actual), physical(cpu, expected)) < tol
                    end
                    before = map(Array, spatial)
                    recovered = SHTns.analys(cfg, spatial...)
                    recovered = components == 1 ? (recovered,) : recovered
                    CUDA.device_synchronize()
                    @test all(eltype(x) === Complex{T} for x in recovered)
                    @test all(Array(x) == y for (x,y) in zip(spatial, before))
                    for (actual, expected) in zip(recovered, h)
                        @test relative(vec(Array(actual)), vec(expected)) < tol
                    end
                    output = map(similar, spatial)
                    SHTns.synth!(cfg, g..., output...)
                    buffers = map(similar, g)
                    SHTns.analys!(cfg, output..., buffers...)
                    CUDA.device_synchronize()
                    for (actual, expected) in zip(buffers, h)
                        @test relative(Array(actual), expected) < tol
                    end
                    wrong = T === Float32 ? Float64 : Float32
                    bad_spectra = map(x -> CuArray(Complex{wrong}.(Array(x))), g)
                    bad_spatial = map(x -> CuArray(wrong.(Array(x))), spatial)
                    @test_throws ArgumentError SHTns.synth(cfg, bad_spectra...)
                    @test_throws ArgumentError SHTns.analys(cfg, bad_spatial...)
                    @test_throws ArgumentError SHTns.synth!(cfg, bad_spectra..., bad_spatial...)
                    @test_throws ArgumentError SHTns.analys!(cfg, bad_spatial..., bad_spectra...)
                end
            end
        end
        GC.gc(); CUDA.reclaim()
    end
end


@testset "CUDA public GPU constructor" begin
    for T in (Float32, Float64)
        cfg = SHTnsCfg(21, 21, 1, 32, 64;
            shtype=SHTns.QuickInit(; gpu=true, float32=T === Float32))
        q = zeros(Complex{T}, nlm(cfg)); q[1] = 1; q[2] = 0.5
        data = SHTns.synth(cfg, CuArray(q))
        recovered = SHTns.analys(cfg, data)
        CUDA.device_synchronize()
        @test relative(Array(recovered), q) < (T === Float32 ? 5e-4 : 5e-10)
    end
end
