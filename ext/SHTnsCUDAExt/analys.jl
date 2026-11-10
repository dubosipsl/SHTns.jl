function analys(cfg::SHTnsCfg, v::CuArray{T}) where {T<:Union{Float32,Float64}}
    check_precision(cfg, T)
    @assert cfg.nlat != 0
    qlm = cfg.howmany > 1 ? CuMatrix{complex(T)}(undef, nlm(cfg), cfg.howmany) : CuVector{complex(T)}(undef, nlm(cfg))
    analys!(cfg, copy(v), qlm)
    return qlm
end

function analys(cfg::SHTnsCfg, utheta::CuArray{T}, uphi::CuArray{T}) where {T<:Union{Float32,Float64}}
    check_precision(cfg, T)
    @assert cfg.nlat != 0
    slm = cfg.howmany > 1 ? CuMatrix{complex(T)}(undef, nlm(cfg), cfg.howmany) : CuVector{complex(T)}(undef, nlm(cfg))
    tlm = cfg.howmany > 1 ? CuMatrix{complex(T)}(undef, nlm(cfg), cfg.howmany) : CuVector{complex(T)}(undef, nlm(cfg))
    analys!(cfg, copy(utheta), copy(uphi), slm, tlm)
    return slm, tlm
end

function analys(cfg::SHTnsCfg, ur::CuArray{T}, utheta::CuArray{T}, uphi::CuArray{T}) where {T<:Union{Float32,Float64}}
    check_precision(cfg, T)
    @assert cfg.nlat != 0
    qlm = cfg.howmany > 1 ? CuMatrix{complex(T)}(undef, nlm(cfg), cfg.howmany) : CuVector{complex(T)}(undef, nlm(cfg))
    slm = cfg.howmany > 1 ? CuMatrix{complex(T)}(undef, nlm(cfg), cfg.howmany) : CuVector{complex(T)}(undef, nlm(cfg))
    tlm = cfg.howmany > 1 ? CuMatrix{complex(T)}(undef, nlm(cfg), cfg.howmany) : CuVector{complex(T)}(undef, nlm(cfg))
    analys!(cfg, copy(ur), copy(utheta), copy(uphi), qlm, slm, tlm)
    return qlm, slm, tlm
end

function analys!(cfg::SHTnsCfg, v::CuArray{F}, qlm::CuArray{Complex{F}}) where {F<:Union{Float32,Float64}}
    check_precision(cfg, F)
    return cu_spat_to_SH(cfg.cfg, v, qlm, cfg.lmax)
end


function analys!(cfg::SHTnsCfg, utheta::CuArray{F}, uphi::CuArray{F}, slm::CuArray{Complex{F}}, tlm::CuArray{Complex{F}}) where {F<:Union{Float32,Float64}}
    check_precision(cfg, F)
    return cu_spat_to_SHsphtor(cfg.cfg, utheta, uphi, slm, tlm, cfg.lmax)
end

function analys!(cfg::SHTnsCfg, ur::CuArray{F}, utheta::CuArray{F}, uphi::CuArray{F}, qlm::CuArray{Complex{F}}, slm::CuArray{Complex{F}}, tlm::CuArray{Complex{F}}) where {F<:Union{Float32,Float64}}
    check_precision(cfg, F)
    return cu_spat_to_SHqst(cfg.cfg, ur, utheta, uphi, qlm, slm, tlm, cfg.lmax)
end


# end

#complex to complex not available for CUDA (status: SHTns v3.7)

# function analys!(cfg::SHTnsCfg, v::CuArray{ComplexF64}, qlm::CuVector{ComplexF64})
#     return cu_spat_cplx_to_SH(cfg.cfg, v, qlm)
# end

# function analys!(cfg::SHTnsCfg, utheta::T, uphi::T, slm::CuVector{ComplexF64}, tlm::CuVector{ComplexF64}) where {T<:CuArray{ComplexF64}}
#     return cu_spat_cplx_to_SHsphtor(cfg.cfg, utheta, uphi, slm, tlm)
# end

# function analys!(cfg::SHTnsCfg, ur::T, utheta::T, uphi::T, qlm::CuVector{ComplexF64}, slm::CuVector{ComplexF64}, tlm::CuVector{ComplexF64}) where {T<:CuArray{ComplexF64}}
#     return cu_spat_cplx_to_SHqst(cfg.cfg, ur, utheta, uphi, qlm, slm, tlm)
# end
