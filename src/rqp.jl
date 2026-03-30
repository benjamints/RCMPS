function ϕnWsys!(du, u, p, x, affine=true)
    Q, R, FP, dims, dQ, dR, FPW, k, solρ = p
    ϕnsys!(du, u, (Q, R, FPW, dims), x, affine)

    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    for n in axes(ρ, 3)
        dρ[:, :, n] .+= -im * k * ρ[:, :, n]
    end
    if affine
        ρ0 = reshape(solρ(x), dims)
        j = J(x)

        for n in axes(ρ, 3)
            dρ[:, :, n] .+= dQ * ρ0[:, :, n] + dR * ρ0[:, :, n] * R'
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
    Q, R, A, FP, dims, dQ, dR, FPW, dA, k, solρ = p
    a11sys!(du, u, (Q, R, A, FPW, dims), x, affine)

    dρ = reshape(du, dims)
    ρ = reshape(u, dims)
    for n in 1:3
        dρ[:, :, n] .+= -im * k * ρ[:, :, n]
    end
    if affine
        ρ0 = reshape(solρ(x), dims)
        j = J(x)

        for n in 1:3
            dρ[:, :, n] .+= dQ * ρ0[:, :, n] + dR * ρ0[:, :, n] * R'
        end
        dρ[:, :, 2] .+= j * dA * FP
        dρ[:, :, 3] .+= j * dA * ρ0[:, :, 1]
    end
    nothing
end

function solveFPW(ψ::LeftGaugedRCMPS, k, W::Array{ComplexF64,2})
    b = ψ.R' * W * ψ.rFP - W * ψ.rFP * ψ.R'
    rFPW, _ = linsolve(u -> ψ.Q * u + u * ψ.Q' + ψ.R * u * ψ.R' - im * k * u + ψ.rFP * tr(u), b)
    return rFPW
end

function ϕnH(ψ::LeftGaugedRCMPS, n::Int64, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψ, k, W)

    dim = (bonddim(ψ), bonddim(ψ), n)

    dQ = -ψ.R' * W
    dR = W

    solρW, solOW = integrateSol(ϕnWsys!, dim, (ψ.Q, ψ.R, ψ.rFP, dim, dQ, dR, rFPW, k, solρ), (ψ.Q', ψ.R', ψ.lFP, dim, dQ', dR', 0 * rFPW, k, solO))

    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :, n))

    M, ee = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψ.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        #m is # b derivatives on O
        sum .+= -ψ.R * (OxW[:, :, n]' * ψ.rFP + Ox[:, :, n]' * rFPW) + OxW[:, :, n]' * ψ.R * ψ.rFP +
                Ox[:, :, n]' * dR * ψ.rFP + Ox[:, :, n]' * ψ.R * rFPW
        for m in 1:n-1
            sum .+= binomial(n, m) * (-ψ.R * (OxW[:, :, m]' * ρx[:, :, n-m] + Ox[:, :, m]' * ρxW[:, :, n-m]) +
                                      OxW[:, :, m]' * ψ.R * ρx[:, :, n-m] + Ox[:, :, m]' * dR * ρx[:, :, n-m] +
                                      Ox[:, :, m]' * ψ.R * ρxW[:, :, n-m])
        end
        sum .+= dR * ρx[:, :, n]

        if n == 1
            sum .+= n * j * rFPW
        else
            sum .+= n * j * (OxW[:, :, n-1]' * ψ.rFP + Ox[:, :, n-1]' * rFPW)
        end
        for m in 1:n-2
            sum .+= n * j * binomial(n - 1, m) * (OxW[:, :, m]' * ρx[:, :, n-1-m] + Ox[:, :, m]' * ρxW[:, :, n-1-m])
        end
        if n > 1
            sum .+= n * j * ρxW[:, :, n-1]
        end

        sum -= VEV * W * ψ.rFP
        return sum
    end
    # println("ee = $ee")
    return M
end

function aZH(ψ::LeftGaugedRCMPS, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψ, k, W)
    dim = (bonddim(ψ), bonddim(ψ), 3)

    dQ = -ψ.R' * W
    dR = W
    A = ψ.R
    dA = W
    #Q, R, A, FP, dims, dQ, dR, FPW, dA, k, solρ = p
    solρW, solOW = integrateSol(a11Wsys!, dim, (ψ.Q, ψ.R, A, ψ.rFP, dim, dQ, dR, rFPW, dA, k, solρ), (ψ.Q', ψ.R', A', ψ.lFP, dim, dQ', dR', 0 * rFPW, dA', k, solO))

    # return solρW, solOW
    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :, 3))

    M, ee = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψ.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        sum .+= -ψ.R * (OxW[:, :, 3]' * ψ.rFP + Ox[:, :, 3]' * rFPW +
                        OxW[:, :, 2]' * ρx[:, :, 1] + Ox[:, :, 2]' * ρxW[:, :, 1] +
                        OxW[:, :, 1]' * ρx[:, :, 2] + Ox[:, :, 1]' * ρxW[:, :, 2] +
                        ρxW[:, :, 3])
        sum .+= OxW[:, :, 3]' * ψ.R * ψ.rFP + Ox[:, :, 3]' * dR * ψ.rFP + Ox[:, :, 3]' * ψ.R * rFPW +
                OxW[:, :, 2]' * ψ.R * ρx[:, :, 1] + Ox[:, :, 2]' * dR * ρx[:, :, 1] + Ox[:, :, 2]' * ψ.R * ρxW[:, :, 1] +
                OxW[:, :, 1]' * ψ.R * ρx[:, :, 2] + Ox[:, :, 1]' * dR * ρx[:, :, 2] + Ox[:, :, 1]' * ψ.R * ρxW[:, :, 2] +
                dR * ρx[:, :, 3] + ψ.R * ρxW[:, :, 3]
        sum .+= j * (OxW[:, :, 2]' * ψ.rFP + Ox[:, :, 2]' * rFPW +
                     ρxW[:, :, 2])

        sum -= VEV * W * ψ.rFP
        return sum
    end
    # println("ee = $ee")
    return M
end

function aYH(ψ::LeftGaugedRCMPS, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψ, k, W)
    dim = (bonddim(ψ), bonddim(ψ), 3)

    dQ = -ψ.R' * W
    dR = W
    A = CC(ψ.Q, ψ.R)
    dA = -CC(ψ.R' * W, ψ.R) + CC(ψ.Q, W) - im * k * dR
    #Q, R, A, FP, dims, dQ, dR, FPW, dA, k, solρ = p
    solρW, solOW = integrateSol(a11Wsys!, dim, (ψ.Q, ψ.R, A, ψ.rFP, dim, dQ, dR, rFPW, dA, k, solρ), (ψ.Q', ψ.R', A', ψ.lFP, dim, dQ', dR', 0 * rFPW, dA', k, solO))

    # return solρW, solOW
    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :, 3))

    M, ee = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        sum = zero(ψ.K)
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        sum .+= -ψ.R * (OxW[:, :, 3]' * ψ.rFP + Ox[:, :, 3]' * rFPW +
                        OxW[:, :, 2]' * ρx[:, :, 1] + Ox[:, :, 2]' * ρxW[:, :, 1] +
                        OxW[:, :, 1]' * ρx[:, :, 2] + Ox[:, :, 1]' * ρxW[:, :, 2] +
                        ρxW[:, :, 3])
        sum .+= OxW[:, :, 3]' * ψ.R * ψ.rFP + Ox[:, :, 3]' * dR * ψ.rFP + Ox[:, :, 3]' * ψ.R * rFPW +
                OxW[:, :, 2]' * ψ.R * ρx[:, :, 1] + Ox[:, :, 2]' * dR * ρx[:, :, 1] + Ox[:, :, 2]' * ψ.R * ρxW[:, :, 1] +
                OxW[:, :, 1]' * ψ.R * ρx[:, :, 2] + Ox[:, :, 1]' * dR * ρx[:, :, 2] + Ox[:, :, 1]' * ψ.R * ρxW[:, :, 2] +
                dR * ρx[:, :, 3] + ψ.R * ρxW[:, :, 3]
        sum .+= j * (CC(ψ.Q', OxW[:, :, 2]' * ψ.rFP + Ox[:, :, 2]' * rFPW) - ψ.R * CC(OxW[:, :, 2]' * ψ.rFP + Ox[:, :, 2]' * rFPW, ψ.R') +
                     CC(ψ.Q', ρxW[:, :, 2]) - ψ.R * CC(ρxW[:, :, 2], ψ.R'))
        sum .+= im * k * j * (OxW[:, :, 2]' * ψ.rFP + Ox[:, :, 2]' * rFPW + ρxW[:, :, 2])

        sum -= VEV * W * ψ.rFP
        return sum
    end
    # println("ee = $ee")
    return M
end


function expϕWsys!(du, u, p, x, affine=true)
    Q, R, β, dims, dQ, dR, k, solρ = p
    expϕsys!(du, u, (Q, R, β, dims), x, affine)

    dρ = reshape(du, dims)
    ρ = reshape(u, dims)

    dρ[:, :] .+= -im * k * ρ[:, :]

    if affine
        ρ0 = reshape(solρ(x), dims)
        j = J(x)
        dρ[:, :] .+= dQ * ρ0[:, :] + dR * ρ0[:, :] * R' + im * β * j * dR * ρ0[:, :]
    end

    nothing
end

function expϕH(ψ::LeftGaugedRCMPS, β::Float64, k::Float64, solρ, solO, W::Array{ComplexF64,2})
    rFPW = solveFPW(ψ, k, W)

    dim = (bonddim(ψ), bonddim(ψ))

    dQ = -ψ.R' * W
    dR = W

    solρW, solOW = integrateSol(expϕWsys!, dim, (ψ.Q, ψ.R, β, dim, dQ, dR, k, solρ), (ψ.Q', ψ.R', -β, dim, dQ', dR', k, solO), rFPW)

    VEV = tr(view(reshape(solρ(integration_limit), dim), :, :))

    M, _ = quadde(-integration_limit, 0, integration_limit; rtol=int_tol) do x
        ρx = reshape(solρ(x), dim)
        Ox = reshape(solO(-x), dim)
        ρxW = reshape(solρW(x), dim)
        OxW = reshape(solOW(-x), dim)
        j = J(x)

        return -ψ.R * OxW' * ρx - ψ.R * Ox' * ρxW + OxW' * ψ.R * ρx + Ox' * dR * ρx + Ox' * ψ.R * ρxW +
               +im * β * j * (OxW' * ρx + Ox' * ρxW) - VEV * W * ψ.rFP
    end

    return M
end