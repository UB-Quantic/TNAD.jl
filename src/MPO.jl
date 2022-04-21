using TensorOperations

import Base: zeros
import Random: rand
import LinearAlgebra: norm

struct MPO{T<:Real}
    t::Vector{Array{T}} # in(up)-out(down)-left-right
    χ::UInt
end

function zeros(...)
    throw()
end


function norm(A::MPO, p::Real=2)
    for t in A.t
        @tensor C[u, d, l, r] := A[] * A[]
    end
    ...
end