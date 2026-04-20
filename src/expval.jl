# ϕ^n VEV
function ϕnsys!(du, u, p, x, affine=true)
    QL, RL, QR, RR, FP, dims = p
    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    j = J(x)

    for n in axes(ρ, 3)
        dρ[:, :, n] .= QL * ρ[:, :, n] + ρ[:, :, n] * QR' + RL * ρ[:, :, n] * RR'
        if n == 1 && affine
            dρ[:, :, n] .+= j * (RL * FP + FP * RR')
        elseif n > 1
            dρ[:, :, n] .+= n * j * (RL * ρ[:, :, n-1] + ρ[:, :, n-1] * RR')
        end
    end
    nothing
end

function ϕnVEV(ψ::LeftGaugedRCMPS, n::Int64)
    dim = (bonddim(ψ), bonddim(ψ), n)
    v0 = zeros(ComplexF64, dim)

    p = (ψ.Q, ψ.R, ψ.Q, ψ.R, ψ.rFP, dim)
    sol = integrateODE(ϕnsys!, vec(v0), p)
    ρ = reshape(sol, dim)
    return tr(ρ[:, :, end])
end

function ϕnDer(ψ::LeftGaugedRCMPS, n::Int64)
    dim = (bonddim(ψ), bonddim(ψ), n)
    solρ, solO = integrateSol(ϕnsys!, dim, (ψ.Q, ψ.R, ψ.Q, ψ.R, ψ.rFP, dim), (ψ.Q', ψ.R', ψ.Q', ψ.R', ψ.lFP, dim))

    # v0 = zeros(ComplexF64, dim)

    # solρ = integrateODE(ϕnsys!, vec(v0), (ψ.Q, ψ.R, ψ.rFP, dim), true)
    # solO = integrateODE(ϕnsys!, vec(v0), (ψ.Q', ψ.R', ψ.lFP, dim), true)

    M, _ = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψ.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        j = J(x)

        sum += ψ.rFP * (ψ.R' * Ox[:, :, n]' - Ox[:, :, n]' * ψ.R')
        for m in 1:n-1
            sum += binomial(n, m) * ρx[:, :, m] * (ψ.R' * Ox[:, :, n-m]' - Ox[:, :, n-m]' * ψ.R')
        end
        # sum += -ρx[n, :, :] * ψ.R' + ρx[n, :, :] * ψ.R'

        if n == 1
            sum += n * j * ψ.rFP
        else
            sum += n * j * ψ.rFP * Ox[:, :, n-1]'
        end
        for m in 1:n-2
            sum += n * j * binomial(n - 1, m) * ρx[:, :, m] * Ox[:, :, n-1-m]'
        end
        if n > 1
            sum += n * j * ρx[:, :, n-1]
        end
        return sum
    end
    # @show ee
    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :, n))

    return VEV, M
end

# H_fb
# Order (n,m) = 01, 10, 11
function a11sys!(du, u, p, x, affine=true)
    QL, RL, AL, QR, RR, AR, FP, dims = p
    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    j = J(x)

    for n in 1:3
        dρ[:, :, n] .= QL * ρ[:, :, n] + ρ[:, :, n] * QR' + RL * ρ[:, :, n] * RR'
        if n == 1 && affine
            dρ[:, :, n] .+= j * FP * AR'
        elseif n == 2 && affine
            dρ[:, :, n] .+= j * AL * FP
        elseif n > 2
            dρ[:, :, n] .+= j * AL * ρ[:, :, 1] + j * ρ[:, :, 2] * AR'
        end
    end
    nothing
end


function aZDer(ψ::LeftGaugedRCMPS)
    dim = (bonddim(ψ), bonddim(ψ), 3)
    solρ, solO = integrateSol(a11sys!, dim, (ψ.Q, ψ.R, ψ.R, ψ.Q, ψ.R, ψ.R, ψ.rFP, dim), (ψ.Q', ψ.R', ψ.R', ψ.Q', ψ.R', ψ.R', ψ.lFP, dim))

    M, ee = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψ.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        j = J(x)

        sum += ρx[:, :, 2] * CC(ψ.R', Ox[:, :, 1]')
        sum += ρx[:, :, 1] * CC(ψ.R', Ox[:, :, 2]')
        sum += ψ.rFP * CC(ψ.R', Ox[:, :, 3]')
        sum += j * (ρx[:, :, 1] + ψ.rFP * Ox[:, :, 1]')
        return sum
    end
    # @show ee
    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :, 3))

    return VEV, M
end

function aYDer(ψ::LeftGaugedRCMPS)
    A = CC(ψ.Q, ψ.R)

    dim = (bonddim(ψ), bonddim(ψ), 3)
    solρ, solO = integrateSol(a11sys!, dim, (ψ.Q, ψ.R, A, ψ.Q, ψ.R, A, ψ.rFP, dim), (ψ.Q', ψ.R', A', ψ.Q', ψ.R', A', ψ.lFP, dim))
    # v0 = zeros(ComplexF64, dim)

    # solρ = integrateODE(a11sys!, vec(v0), (ψ.Q, ψ.R, A, ψ.rFP, dim), true)
    # solO = integrateODE(a11sys!, vec(v0), (ψ.Q', ψ.R', A', ψ.lFP, dim), true)


    M, ee = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψ.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        j = J(x)

        sum += ρx[:, :, 2] * CC(ψ.R', Ox[:, :, 1]')
        sum += ρx[:, :, 1] * CC(ψ.R', Ox[:, :, 2]')
        sum += ψ.rFP * CC(ψ.R', Ox[:, :, 3]')
        sum += j * (CC(ρx[:, :, 1], ψ.R) + CC(ψ.rFP * Ox[:, :, 1]', ψ.R)) * ψ.R'
        sum += j * (CC(ρx[:, :, 1], ψ.Q) + CC(ψ.rFP * Ox[:, :, 1]', ψ.Q))
        return sum
    end
    # @show ee
    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :, 3))

    return VEV, M
end

function a11VEV(ψ::LeftGaugedRCMPS, A::Array{ComplexF64,2})
    dim = (bonddim(ψ), bonddim(ψ), 3)
    v0 = zeros(ComplexF64, dim)

    p = (ψ.Q, ψ.R, A, ψ.Q, ψ.R, A, ψ.rFP, dim)
    sol = integrateODE(a11sys!, vec(v0), p)
    ρ = reshape(sol, dim)
    return tr(ρ[:, :, end])
end

#vertex op

function expϕsys!(du, u, p, x, affine=true)
    QL, RL, QR, RR, β, dims = p
    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    j = J(x)

    dρ[:, :] .= QL * ρ[:, :] + ρ[:, :] * QR' + RL * ρ[:, :] * RR' +
                im * β * j * (RL * ρ[:, :] + ρ[:, :] * RR')

    nothing
end

function expϕVEV(ψ::LeftGaugedRCMPS, β::Float64)
    dim = (bonddim(ψ), bonddim(ψ))
    v0 = ψ.rFP

    p = (ψ.Q, ψ.R, ψ.Q, ψ.R, β, dim)
    sol = integrateODE(expϕsys!, vec(v0), p)
    ρ = reshape(sol, dim)
    return tr(ρ[:, :])
end

function expϕDer(ψ::LeftGaugedRCMPS, β::Float64)
    dim = (bonddim(ψ), bonddim(ψ))
    solρ, solO = integrateSol(expϕsys!, dim, (ψ.Q, ψ.R, ψ.Q, ψ.R, β, dim), (ψ.Q', ψ.R', ψ.Q', ψ.R', -β, dim), ψ.rFP, ψ.lFP)

    M, _ = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        j = J(x)

        return ρx * CC(ψ.R', Ox') + im * β * j * ρx * Ox'
    end

    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :))

    return VEV, M
end