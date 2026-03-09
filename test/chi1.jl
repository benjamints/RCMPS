using Test
using LinearAlgebra

ω(k) = sqrt(k^2 + 1)

@testset "χ = 1 tests:" begin
    @testset "VEVs" begin
        ψ = LeftGaugedRCMPS(1)
        for n in 1:5

            VEVϕn = ϕnVEV(ψ, n)
            VEVϕn1 = (sqrt(2) * real(ψ.R[1]))^n

            @test VEVϕn ≈ VEVϕn1 atol = 1e-2
        end

        VEVa11 = a11VEV(ψ, ψ.R)
        VEVa111 = abs(ψ.R[1])^2 / 2
        @test VEVa11 ≈ VEVa111 atol = 1e-2

        for β = [0.5, 1.0, 2.0]
            eVEV = cis(sqrt(2) * β * real(ψ.R[1]))
            eVEV1 = expϕVEV(ψ, β)
            @test eVEV ≈ eVEV1 atol = 1e-2
        end
    end

    @testset "Derivatives" begin
        ψ = LeftGaugedRCMPS(1)
        for n in 1:5
            VEVϕn, Derϕn = ϕnDer(ψ, n)
            VEVϕn1 = (sqrt(2) * real(ψ.R[1]))^n
            Derϕn1 = n * (sqrt(2) * real(ψ.R[1]))^(n - 1) / sqrt(2)

            @test VEVϕn ≈ VEVϕn1 atol = 1e-2
            @test Derϕn[1] ≈ Derϕn1 atol = 1e-2
        end

        VEVa11, Dera11 = aZDer(ψ)
        VEVa111 = abs(ψ.R[1])^2 / 2
        Dera111 = conj(ψ.R[1]) / 2
        @test VEVa11 ≈ VEVa111 atol = 1e-2
        @test Dera11[1] ≈ Dera111 atol = 1e-2

        for β = [0.5, 1.0, 2.0]
            eDer = im * β / sqrt(2) * cis(sqrt(2) * β * real(ψ.R[1]))
            _, eDer1 = expϕDer(ψ, β)
            @test eDer ≈ eDer1[1] atol = 1e-2
        end
    end

    @testset "Effective Hamiltonian" begin
        ψ = LeftGaugedRCMPS(1)
        W = ones(ComplexF64, 1, 1)

        for k = [0.0, 0.4, 1.0]
            for n in 1:5
                dim = (bonddim(ψ), bonddim(ψ), n)
                solρ, solO = integrateSol(ϕnsys!, dim, (ψ.Q, ψ.R, ψ.rFP, dim), (ψ.Q', ψ.R', ψ.lFP, dim))
                Wp = ϕnH(ψ, n, k, solρ, solO, W)

                @test Wp[1] ≈ n * (n - 1) / (2 * ω(k)) * 2^(n / 2 - 1) * real(ψ.R[1])^(n - 2) atol = 1e-2
            end

            dim = (bonddim(ψ), bonddim(ψ), 3)
            solρ, solO = integrateSol(a11sys!, dim, (ψ.Q, ψ.R, ψ.R, ψ.rFP, dim), (ψ.Q', ψ.R', ψ.R', ψ.lFP, dim))
            Wp = aZH(ψ, k, solρ, solO, W)

            @test Wp[1] ≈ 1 / (2 * ω(k)) atol = 1e-2

            for β = [0.5, 1.0, 2.0]
                dim = (bonddim(ψ), bonddim(ψ))
                solρ, solO = integrateSol(expϕsys!, dim, (ψ.Q, ψ.R, β, dim), (ψ.Q', ψ.R', -β, dim), ψ.rFP, ComplexF64.(I(bonddim(ψ))))
                Wp = expϕH(ψ, β, k, solρ, solO, W)

                @test Wp[1] ≈ -β^2 / (2 * ω(k)) * cis(sqrt(2) * β * real(ψ.R[1])) atol = 1e-2
            end
        end
    end
end
