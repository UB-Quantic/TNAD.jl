module TNAD

include("MPS.jl")
include("MPO.jl")
include("FeatureMap.jl")

export MPS, physical_dim
export SpacedMPO
export Φ, ϕtrig, ϕfourier

end # module
