using TensorOperations

import Base: zeros
import Random: rand
import LinearAlgebra: norm

struct SpacedMPO{T}
    tensors::Vector{Array{T}} # in(up)-out(down)-left-right
end

SpacedMPO(tensors::Vector{Array{T,N}}) where {T,N} = SpacedMPO{T}(tensors)

function norm(A::MPO, p::Real=2)
    for t in A.t
        @tensor C[u, d, l, r] := A[] * A[]
    end
    ...
end