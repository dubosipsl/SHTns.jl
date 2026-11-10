module SHTnsCUDAExt

using CUDA
using SHTns

import SHTns: libshtns, shtns_cfg
import SHTns: synth, synth!, analys, analys!

__init__() = @assert CUDA.functional()

function check_precision(cfg, ::Type{T}) where T
    @assert cfg.shtype.gpu
    expected = cfg.shtype.float32 ? Float32 : Float64
    T === expected || throw(ArgumentError("CUDA plan expects $expected data, got $T"))
    return nothing
end

include("sht.jl")
include("synth.jl")
include("analys.jl")

end #module