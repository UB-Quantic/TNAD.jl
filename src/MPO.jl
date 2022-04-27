using Random
using TensorOperations

import Base: eltype, rand
import LinearAlgebra: norm

struct SpacedMPO{T}
    tensors::Vector{Array{T}} # in(up)-out(down)-left-right
end

SpacedMPO(tensors::Vector{Array{T,N}}) where {T,N} = SpacedMPO{T}(tensors)

eltype(::SpacedMPO{T}) where {T} = T
length(m::SpacedMPO) = length(m.tensors)
physical_dim(m::SpacedMPO) = size(first(m.tensors), 1)

struct MPOSampler{T}
    n
    p
    χ
    s
end

MPOSampler(n, p, χ, s=1) = MPOSampler{Float64}(n, p, χ, s)
MPOSampler(::Type{T}, n, p, χ, s=1) where {T} = MPOSampler{T}(n, p, χ, s)

"""
    SpacedMPO(n::Integer, p::Integer, χ::Integer[, s::Integer])

Return a `Random.Sampler` for a randomly sampling `SpacedMPO`s of the given parameters.
"""
SpacedMPO(n::Integer, p::Integer, χ::Integer, s::Integer=1) = SpacedMPO(Float64, n, p, χ, s)
SpacedMPO(::Type{T}, n::Integer, p::Integer, χ::Integer, s::Integer=1) where {T} = MPOSampler{T}(n, p, χ, s)

"""
    rand(::Type{SpacedMPO{T}}, n, p, χ[, spacing]])

Generate a `SpacedMPO` composed of `n` tensors, physical dimension `p`, bond dimension `χ` and spacing `s`.
"""
function rand(rng::AbstractRNG, sampler::Random.SamplerTrivial{MPOSampler{T}}) where {T}
    n = sampler[].n
    p = sampler[].p
    χ = sampler[].χ
    s = sampler[].s

    # W = ⊗ ℝ^q
    q = (n - 1) ÷ s + 1

    # index of start of stationary region
    t = min(ceil(Int, 1 / 2 * log(p, χ)), n ÷ 2)

    tensors = [
        # tensors in the left transient
        [rand(rng, T, p, p, p^(2 * (i - 1)), p^(2 * i)) for i ∈ 1:t-1]

        # start of stationary region
        [rand(rng, T, p, p, p^(2 * (t - 1)), χ)]

        # tensors in the stationary region
        [rand(rng, T, p, p, χ, χ) for _ ∈ t+1:n-t]

        # end of stationary region
        [rand(rng, T, p, p, χ, p^(2 * (t - 1)))]

        # tensors in the right transient
        [rand(rng, T, p, p, p^(2 * i), p^(2 * (i - 1))) for i ∈ t-1:-1:1]
    ]

    SpacedMPO(tensors)
end


"""
    norm(X::SpacedMPO, p::Real=2)

Calculates the 2-norm of `X` by computing `tr(X * adjoint(X))`.
"""
function norm(X::SpacedMPO, p::Real=2)
    if p != 2
        error("norm(; p=$p) not implemented")
    end

    A = []
    for T ∈ X.tensors
        Th = conj(T)
        @tensor U[left1, left2, right1, right2] := T[up, down, left1, right1] * Th[up, down, left2, right2]
        push!(A, U)
    end

    B = popfirst!(A)
    for T ∈ A
        @tensor U[left1, left2, right1, right2] := B[left1, left2, shared1, shared2] * T[shared1, shared2, right1, right2]
        B = U
    end

    only(B)
end