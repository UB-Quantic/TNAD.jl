ϕtrig(; k) = x -> ϕtrig(x; k=k)
ϕtrig(x; k=1) = 1 / √k * [f(π / 2^k * x) for f ∈ (cos, sin) for i ∈ 1:k]

ϕfourier(; p) = x -> ϕfourier(x; p=p)
ϕfourier(x; p=2) = 1 / p * [abs(sum(cis(2 * π * k * ((p - 1) * x - j) / p) for k ∈ 0:p-1)) for j ∈ 0:p-1]

Φ(; ϕ) = x -> Φ(x; ϕ=ϕ)
Φ(𝐱::Vector; ϕ=ϕfourier) = map(ϕ, 𝐱)