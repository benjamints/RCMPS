function ϕnWsys!(du, u, p, x, affine=true)
    QL, RL, QR, RR, FP, dims, dQ, dR, FPW, k, solρ = p
    ϕnsys!(du, u, (QL, RL, QR, RR, FPW, dims), x, affine)

    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    for n in axes(ρ, 3)
        dρ[:, :, n] .+= -im * k * ρ[:, :, n]
    end
    if affine
        ρ0 = reshape(solρ(x), dims)
        j = J(x)

        for n in axes(ρ, 3)
            dρ[:, :, n] .+= dQ * ρ0[:, :, n] + dR * ρ0[:, :, n] * RR'
            if n == 1
                dρ[:, :, n] .+= n * j * dR * FP
            else
                dρ[:, :, n] .+= n * j * dR * ρ0[:, :, n-1]
            end
        end
    end
    nothing
end

function a11Wsys!(du, u, p, x, affine=true)
    QL, RL, AL, QR, RR, AR, FP, dims, dQ, dR, dA, FPW, k, solρ = p
    a11sys!(du, u, (QL, RL, AL, QR, RR, AR, FPW, dims), x, affine)

    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    for n in 1:3
        dρ[:, :, n] .+= -im * k * ρ[:, :, n]
    end
    if affine
        ρ0 = reshape(solρ(x), dims)
        j = J(x)

        for n in 1:3
            dρ[:, :, n] .+= dQ * ρ0[:, :, n] + dR * ρ0[:, :, n] * RR'
        end
        dρ[:, :, 2] .+= j * dA * FP
        dρ[:, :, 3] .+= j * dA * ρ0[:, :, 1]
    end
    nothing
end

solveFPW(ψL::LeftGaugedRCMPS, k, W::Array{ComplexF64,2}) = solveFPW(ψL, ψL, k, W, true)

function solveFPW(ψL::LeftGaugedRCMPS, ψR::LeftGaugedRCMPS, k, W::Array{ComplexF64,2}, reg=false)
    b = ψL.R' * W * ψR.rFP - W * ψR.rFP * ψR.R'
    if reg
        rFPW, _ = linsolve(u -> ψL.Q * u + u * ψR.Q' + ψL.R * u * ψR.R' - im * k * u + ψL.rFP * tr(u), b)
    else
        rFPW, _ = linsolve(u -> ψL.Q * u + u * ψR.Q' + ψL.R * u * ψR.R' - im * k * u, b)
    end
    return rFPW
end

ϕnH(ψL::LeftGaugedRCMPS, n::Int64, k::Float64, solρ, solO, W::Array{ComplexF64,2}) = ϕnH(ψL, ψL, n, k, solρ, solO, W)

function ϕnH(ψL::LeftGaugedRCMPS, ψR::LeftGaugedRCMPS, n::Int64, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψL, ψR, k, W)

    dim = (bonddim(ψR), bonddim(ψR), n)

    dQ = -ψL.R' * W
    dR = W

    solρW, solOW = integrateSol(ϕnWsys!, dim, (ψL.Q, ψL.R, ψR.Q, ψR.R, ψR.rFP, dim, dQ, dR, rFPW, k, solρ), (ψR.Q', ψR.R', ψL.Q', ψL.R', ψL.lFP, dim, dQ', dR', 0 * rFPW, k, solO))

    VEVR = tr(view(reshape(solρ(integration_limit), dim), :, :, n))
    VEVL = tr(view(reshape(solO(integration_limit), dim), :, :, n)' * ψL.rFP)

    M, ee = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψL.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        #m is # b derivatives on O
        sum .+= -ψL.R * (OxW[:, :, n]' * ψR.rFP + Ox[:, :, n]' * rFPW) + OxW[:, :, n]' * ψR.R * ψR.rFP +
                Ox[:, :, n]' * dR * ψR.rFP + Ox[:, :, n]' * ψL.R * rFPW
        for m in 1:n-1
            sum .+= binomial(n, m) * (-ψL.R * (OxW[:, :, m]' * ρx[:, :, n-m] + Ox[:, :, m]' * ρxW[:, :, n-m]) +
                                      OxW[:, :, m]' * ψR.R * ρx[:, :, n-m] + Ox[:, :, m]' * dR * ρx[:, :, n-m] +
                                      Ox[:, :, m]' * ψL.R * ρxW[:, :, n-m])
        end
        sum .+= dR * ρx[:, :, n]

        if n == 1
            sum .+= n * j * rFPW
        else
            sum .+= n * j * (OxW[:, :, n-1]' * ψR.rFP + Ox[:, :, n-1]' * rFPW)
        end
        for m in 1:n-2
            sum .+= n * j * binomial(n - 1, m) * (OxW[:, :, m]' * ρx[:, :, n-1-m] + Ox[:, :, m]' * ρxW[:, :, n-1-m])
        end
        if n > 1
            sum .+= n * j * ρxW[:, :, n-1]
        end

        sum -= (VEVR + VEVL) / 2 * W * ψR.rFP
        return sum
    end
    # println("ee = $ee")
    return M
end

aZH(ψ::LeftGaugedRCMPS, k::Float64, solρ, solO, W::Array{ComplexF64,2}) = aZH(ψ, ψ, k, solρ, solO, W)

function aZH(ψL::LeftGaugedRCMPS, ψR::LeftGaugedRCMPS, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψL, ψR, k, W)
    dim = (bonddim(ψR), bonddim(ψR), 3)

    dQ = -ψL.R' * W
    dR = W
    AL = ψL.R
    AR = ψR.R
    dA = W
    #QL, RL, AL, QR, RR, AR, FP, dims, dQ, dR, dAL, dAR, FPW, k, solρ = p
    solρW, solOW = integrateSol(a11Wsys!, dim, (ψL.Q, ψL.R, AL, ψR.Q, ψR.R, AR, ψR.rFP, dim, dQ, dR, dA, rFPW, k, solρ), (ψR.Q', ψR.R', AR', ψL.Q', ψL.R', AL', ψL.lFP, dim, dQ', dR', dA', 0 * rFPW, k, solO))


    VEVR = tr(view(reshape(solρ(integration_limit), dim), :, :, 3))
    VEVL = tr(view(reshape(solO(integration_limit), dim), :, :, 3)' * ψL.rFP)

    M, _ = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψL.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        sum .+= -ψL.R * (OxW[:, :, 3]' * ψR.rFP + Ox[:, :, 3]' * rFPW +
                         OxW[:, :, 2]' * ρx[:, :, 1] + Ox[:, :, 2]' * ρxW[:, :, 1] +
                         OxW[:, :, 1]' * ρx[:, :, 2] + Ox[:, :, 1]' * ρxW[:, :, 2] +
                         ρxW[:, :, 3])
        sum .+= OxW[:, :, 3]' * ψR.R * ψR.rFP + Ox[:, :, 3]' * dR * ψR.rFP + Ox[:, :, 3]' * ψL.R * rFPW +
                OxW[:, :, 2]' * ψR.R * ρx[:, :, 1] + Ox[:, :, 2]' * dR * ρx[:, :, 1] + Ox[:, :, 2]' * ψL.R * ρxW[:, :, 1] +
                OxW[:, :, 1]' * ψR.R * ρx[:, :, 2] + Ox[:, :, 1]' * dR * ρx[:, :, 2] + Ox[:, :, 1]' * ψL.R * ρxW[:, :, 2] +
                dR * ρx[:, :, 3] + ψL.R * ρxW[:, :, 3]
        sum .+= j * (OxW[:, :, 2]' * ψR.rFP + Ox[:, :, 2]' * rFPW +
                     ρxW[:, :, 2])

        sum -= (VEVR + VEVL) / 2 * W * ψR.rFP
        return sum
    end

    return M
end

aYH(ψ::LeftGaugedRCMPS, k::Float64, solρ, solO, W::Array{ComplexF64,2}) = aYH(ψ, ψ, k, solρ, solO, W)

function aYH(ψL::LeftGaugedRCMPS, ψR::LeftGaugedRCMPS, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψL, ψR, k, W)
    dim = (bonddim(ψR), bonddim(ψR), 3)

    dQ = -ψL.R' * W
    dR = W
    AL = CC(ψL.Q, ψL.R)
    AR = CC(ψR.Q, ψR.R)
    dA = -(ψL.R' * W * ψR.R - ψL.R * ψL.R' * W) + (ψL.Q * W - W * ψR.Q) - im * k * dR

    solρW, solOW = integrateSol(a11Wsys!, dim,
        (ψL.Q, ψL.R, AL, ψR.Q, ψR.R, AR, ψR.rFP, dim, dQ, dR, dA, rFPW, k, solρ),
        (ψR.Q', ψR.R', AR', ψL.Q', ψL.R', AL', ψL.lFP, dim, dQ', dR', dA', 0 * rFPW, k, solO))

    VEVR = tr(view(reshape(solρ(integration_limit), dim), :, :, 3))
    VEVL = tr(view(reshape(solO(integration_limit), dim), :, :, 3)' * ψL.rFP)

    M, _ = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψL.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        sum .+= -ψL.R * (OxW[:, :, 3]' * ψR.rFP + Ox[:, :, 3]' * rFPW +
                         OxW[:, :, 2]' * ρx[:, :, 1] + Ox[:, :, 2]' * ρxW[:, :, 1] +
                         OxW[:, :, 1]' * ρx[:, :, 2] + Ox[:, :, 1]' * ρxW[:, :, 2] +
                         ρxW[:, :, 3])
        sum .+= OxW[:, :, 3]' * ψR.R * ψR.rFP + Ox[:, :, 3]' * dR * ψR.rFP + Ox[:, :, 3]' * ψL.R * rFPW +
                OxW[:, :, 2]' * ψR.R * ρx[:, :, 1] + Ox[:, :, 2]' * dR * ρx[:, :, 1] + Ox[:, :, 2]' * ψL.R * ρxW[:, :, 1] +
                OxW[:, :, 1]' * ψR.R * ρx[:, :, 2] + Ox[:, :, 1]' * dR * ρx[:, :, 2] + Ox[:, :, 1]' * ψL.R * ρxW[:, :, 2] +
                dR * ρx[:, :, 3] + ψL.R * ρxW[:, :, 3]
        sum .+= j * ((ψL.Q' * (OxW[:, :, 2]' * ψR.rFP + Ox[:, :, 2]' * rFPW) - (OxW[:, :, 2]' * ψR.rFP + Ox[:, :, 2]' * rFPW) * ψR.Q') -
                     ψL.R * ((OxW[:, :, 2]' * ψR.rFP + Ox[:, :, 2]' * rFPW) * ψR.R' - ψL.R' * (OxW[:, :, 2]' * ψR.rFP + Ox[:, :, 2]' * rFPW)) +
                     (ψL.Q' * ρxW[:, :, 2] - ρxW[:, :, 2] * ψR.Q') - ψL.R * (ρxW[:, :, 2] * ψR.R' - ψL.R' * ρxW[:, :, 2]))
        sum .+= im * k * j * (OxW[:, :, 2]' * ψR.rFP + Ox[:, :, 2]' * rFPW + ρxW[:, :, 2])

        sum -= (VEVR + VEVL) / 2 * W * ψR.rFP
        return sum
    end
    return M
end


function expϕWsys!(du, u, p, x, affine=true)
    QL, RL, QR, RR, β, dims, dQ, dR, k, solρ = p
    expϕsys!(du, u, (QL, RL, QR, RR, β, dims), x, affine)

    dρ = reshape(du, dims)
    ρ = reshape(u, dims)

    dρ[:, :] .+= -im * k * ρ[:, :]

    if affine
        ρ0 = reshape(solρ(x), dims)
        j = J(x)
        dρ[:, :] .+= dQ * ρ0[:, :] + dR * ρ0[:, :] * RR' + im * β * j * dR * ρ0[:, :]
    end

    nothing
end

expϕH(ψL::LeftGaugedRCMPS, β::Float64, k::Float64, solρ, solO, W::Array{ComplexF64,2}) = expϕH(ψL, ψL, β, k, solρ, solO, W)

function expϕH(ψL::LeftGaugedRCMPS, ψR::LeftGaugedRCMPS, β::Float64, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψL, ψR, k, W)

    dim = (bonddim(ψR), bonddim(ψR))

    dQ = -ψL.R' * W
    dR = W

    solρW, solOW = integrateSol(expϕWsys!, dim, (ψL.Q, ψL.R, ψR.Q, ψR.R, β, dim, dQ, dR, k, solρ), (ψR.Q', ψR.R', ψL.Q', ψL.R', -β, dim, dQ', dR', k, solO), rFPW)

    VEVR = tr(view(reshape(solρ(integration_limit), dim), :, :))
    VEVL = tr(view(reshape(solO(integration_limit), dim), :, :)' * ψL.rFP)

    M, _ = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        return -ψL.R * OxW' * ρx - ψL.R * Ox' * ρxW + OxW' * ψR.R * ρx + Ox' * dR * ρx + Ox' * ψL.R * ρxW +
               +im * β * j * (OxW' * ρx + Ox' * ρxW) -
               (VEVR + VEVL) / 2 * W * ψR.rFP
    end

    return M
end