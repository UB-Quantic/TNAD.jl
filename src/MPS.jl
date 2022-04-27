using Random
using TensorOperations

import Base: eltype, rand, length, *

struct MPS{T}
    tensors::Vector{Array{T}} # out(down)-left-right

    function MPS(tensors::Vector{Array{T}}) where {T}
        @assert all(size(first(tensors), 1) .== size.(tensors, 1))
        new{T}(map(x -> reshape(x, size(x)..., Iterators.repeated(1, 3 - ndims(x))...), tensors))
    end
end

MPS(tensors::Vector{Array{T,N}}) where {T,N} = MPS{T}(tensors)

eltype(::MPS{T}) where {T} = T

length(m::MPS) = len(m.tensors)
physical_dim(m::MPS) = size(first(m.tensors), 1)

struct MPSSampler{T}
    n
    p
    χ
end

MPSSampler(n, p, χ) = MPSSampler{Float64}(n, p, χ)
MPSSampler(::Type{T}, n, p, χ) where {T} = MPSSampler{T}(n, p, χ)

"""
	MPS([::Type{T},] n::Integer, p::Integer, χ::Integer)

Return a `Random.Sampler` for randomly sampling `MPS`s of the given parameters.
"""
MPS(n::Integer, p::Integer, χ::Integer) = MPS(Float64, n, p, χ)
MPS(::Type{T}, n::Integer, p::Integer, χ::Integer) where {T} = MPSSampler{T}(n, p, χ)

"""
	rand([::Type{MPS{T}},] n::Integer, p::Integer, χ::Integer)

Generate a `MPS` composed of `n` tensors, physical dimension `p` and bond dimension `χ`.
"""
function rand(rng::Random.AbstractRNG, sampler::Random.SamplerTrivial{MPSSampler{T}}) where {T}
    n = sampler[].n
    p = sampler[].p
    χ = sampler[].χ

    # index of start of stationary region
    t = min(ceil(Int, 1 / 2 * log(p, χ)), n ÷ 2)

    tensors = [
        # tensors in the left transient
        [rand(rng, T, p, p^(2 * (i - 1)), p^(2 * i)) for i ∈ 1:t-1]

        # start of stationary region
        [rand(rng, T, p, p^(2 * (t - 1)), χ)]

        # tensors in the stationary region
        [rand(rng, T, p, χ, χ) for _ ∈ t+1:n-t]

        # end of stationary region
        [rand(rng, T, p, χ, p^(2 * (t - 1)))]

        # tensors in the right transient
        [rand(rng, T, p, p^(2 * i), p^(2 * (i - 1))) for i ∈ t-1:-1:1]
    ]

    MPS(tensors)
end

function *(ψa::MPS, ψb::MPS)
    a = first(ψa.tensors)
    b = first(ψb.tensors)

    @tensor tmp[left_a, left_b, right_a, right_b] := a[inner, left_a, right_a] * b[inner, left_b, right_b]

    for (a, b) ∈ zip(Iterators.drop(ψa.tensors, 1), Iterators.drop(ψb.tensors, 1))
        @tensor tmp[left_a, left_b, right_a, right_b] := tmp[left_a, left_b, aux_a, aux_b] * a[inner, aux_a, right_a] * b[inner, aux_b, right_b]
    end

    dropdims(tmp; dims=tuple(findall(size(tmp) .== 1)...))
end