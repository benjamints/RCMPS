struct LeftGaugedRCMPS
    K::Array{ComplexF64,2}
    Q::Array{ComplexF64,2}
    R::Array{ComplexF64,2}
    rFP::Array{ComplexF64,2}
    lFP::Array{ComplexF64,2}
end

function LeftGaugedRCMPS(χ::Int64)
    K = randn(ComplexF64, χ, χ)
    K .= (K + K') / 2
    R = randn(ComplexF64, χ, χ)
    return LeftGaugedRCMPS(K, R)
end

function LeftGaugedRCMPS(K::Array{ComplexF64,2}, R::Array{ComplexF64,2})
    Q = -1im * K - 0.5 * R' * R
    _, rFP = rightFP(Q, R)
    lFP = I(size(Q, 1))
    return LeftGaugedRCMPS(K, Q, R, rFP, lFP)
end

bonddim(ψ::LeftGaugedRCMPS) = size(ψ.Q, 1)

function rightFP(Q, R)
    vals, vecs, info = eigsolve(u -> Q * u + u * Q' + R * u * R', randn(eltype(Q), size(Q)), 1, :LR)
    @assert info.converged > 0 "Right fixed point did not converge"

    vecs[1] ./= tr(vecs[1])

    return vals[1], vecs[1]
end
