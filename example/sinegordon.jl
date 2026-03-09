using RCMPS
using LinearAlgebra, KrylovKit
using ProgressMeter

retract(ψ, W, α) = (LeftGaugedRCMPS(ψ.K + α * 0.5im * (W' * ψ.R - ψ.R' * W), ψ.R + α * W), W)
inner(ψ, ξ1, ξ2) = real(tr(ξ1' * ξ2 * ψ.rFP))

function fg(ψ::LeftGaugedRCMPS, β)
    ϕ2, Mϕ2 = ϕnDer(ψ, 2)
    eb, Meb = expϕDer(ψ, β)
    _, Mebb = expϕDer(ψ, -β)
    aZ, MaZ = aZDer(ψ)
    aY, MaY = aYDer(ψ)

    f = real(2 * (aZ + aY) - 0.5 * ϕ2 - 1 / β^2 * eb)
    M = 2 * (MaZ + MaY) - 0.5 * Mϕ2 - 1 / β^2 * (Meb + Mebb) / 2

    return f, M' / ψ.rFP
end

function f(ψ::LeftGaugedRCMPS, β)
    ϕ2 = ϕnVEV(ψ, 2)
    eb = expϕVEV(ψ, β)
    aZ = a11VEV(ψ, ψ.R)
    aY = a11VEV(ψ, CC(ψ.Q, ψ.R))

    return real(2 * (aZ + aY) - 0.5 * ϕ2 - 1 / β^2 * eb)
end

function HeffSG(ψ, β, k)
    dim = (bonddim(ψ), bonddim(ψ), 2)
    solρ2, solO2 = integrateSol(ϕnsys!, dim, (ψ.Q, ψ.R, ψ.rFP, dim), (ψ.Q', ψ.R', ψ.lFP, dim))

    dim = (bonddim(ψ), bonddim(ψ))
    solρexpp, solOexpp = integrateSol(expϕsys!, dim, (ψ.Q, ψ.R, β, dim), (ψ.Q', ψ.R', -β, dim), ψ.rFP, ψ.lFP)
    solρexpm, solOexpm = integrateSol(expϕsys!, dim, (ψ.Q, ψ.R, -β, dim), (ψ.Q', ψ.R', β, dim), ψ.rFP, ψ.lFP)

    dim = (bonddim(ψ), bonddim(ψ), 3)
    solρZ, solOZ = integrateSol(a11sys!, dim, (ψ.Q, ψ.R, ψ.R, ψ.rFP, dim), (ψ.Q', ψ.R', ψ.R', ψ.lFP, dim))

    dim = (bonddim(ψ), bonddim(ψ), 3)
    A = CC(ψ.Q, ψ.R)
    solρY, solOY = integrateSol(a11sys!, dim, (ψ.Q, ψ.R, A, ψ.rFP, dim), (ψ.Q', ψ.R', A', ψ.lFP, dim))


    return W -> begin
        Wp = 2 * aZH(ψ, k, solρZ, solOZ, W)
        Wp += 2 * aYH(ψ, k, solρY, solOY, W)
        Wp += -0.5 * ϕnH(ψ, 2, k, solρ2, solO2, W)
        Wp += -1 / β^2 * expϕH(ψ, β, k, solρexpp, solOexpp, W) / 2
        Wp += -1 / β^2 * expϕH(ψ, -β, k, solρexpm, solOexpm, W) / 2

        Wp / ψ.rFP
    end
end

# Gradient decent

b = 0.5
β = sqrt(8 * pi) * b

ψ0 = LeftGaugedRCMPS(2)
E, ψ = gdlinesearch(ψ -> f(ψ, β), ψ -> fg(ψ, β), ψ0, inner, retract; c=1e-2, gradtol=1e-3, maxiter=200, verbose=true)

#%% Quasi-particle Hamiltonian

k = 0.0

Heff = HeffSG(ψ, β, k)

# Dense method
M = zeros(ComplexF64, bonddim(ψ)^2, bonddim(ψ)^2)
W = zeros(ComplexF64, bonddim(ψ), bonddim(ψ))
@showprogress for i = 1:bonddim(ψ)^2
    W[i] = 1.0
    M[:, i] = vec(Heff(W))
    W[i] = 0.0
end
vals = eigvals(M)
println("QP = $(vals[1])")


#%% Or Krylov method
W0 = randn(ComplexF64, bonddim(ψ), bonddim(ψ))
vals, vecs, info = eigsolve(Heff, W0, 1, :SR)
println("QP = $(vals[1])")