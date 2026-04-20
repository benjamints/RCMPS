using Test
using LinearAlgebra

@testset "Effective Hamiltonian Hermiticity" begin
    ψL = LeftGaugedRCMPS(2)
    ψR = LeftGaugedRCMPS(2)
    k = 0.7

    W = randn(ComplexF64, bonddim(ψL), bonddim(ψR))
    Wp = randn(ComplexF64, bonddim(ψL), bonddim(ψR))

    @testset "ϕnH" begin
        n = 2
        dim = (bonddim(ψL), bonddim(ψL), n)
        solρ, solO = integrateSol(ϕnsys!, dim, (ψR.Q, ψR.R, ψR.Q, ψR.R, ψR.rFP, dim), (ψL.Q', ψL.R', ψL.Q', ψL.R', ψL.lFP, dim))

        lhs = tr(Wp' * ϕnH(ψL, ψR, n, k, solρ, solO, W))
        rhs = tr(W' * ϕnH(ψL, ψR, n, k, solρ, solO, Wp))
        @test lhs ≈ conj(rhs) atol = 2e-2 rtol = 2e-2
    end

    @testset "aZH" begin
        dim = (bonddim(ψR), bonddim(ψR), 3)
        solρ, solO = integrateSol(
            a11sys!,
            dim,
            (ψR.Q, ψR.R, ψR.R, ψR.Q, ψR.R, ψR.R, ψR.rFP, dim),
            (ψL.Q', ψL.R', ψL.R', ψL.Q', ψL.R', ψL.R', ψL.lFP, dim),
        )

        lhs = tr(Wp' * aZH(ψL, ψR, k, solρ, solO, W))
        rhs = tr(W' * aZH(ψL, ψR, k, solρ, solO, Wp))
        @test lhs ≈ conj(rhs) atol = 2e-2 rtol = 2e-2
    end

    @testset "aYH" begin
        AL = CC(ψL.Q, ψL.R)
        AR = CC(ψR.Q, ψR.R)
        dim = (bonddim(ψR), bonddim(ψR), 3)
        solρ, solO = integrateSol(
            a11sys!,
            dim,
            (ψR.Q, ψR.R, AR, ψR.Q, ψR.R, AR, ψR.rFP, dim),
            (ψL.Q', ψL.R', AL', ψL.Q', ψL.R', AL', ψL.lFP, dim),
        )

        lhs = tr(Wp' * aYH(ψL, ψR, k, solρ, solO, W))
        rhs = tr(W' * aYH(ψL, ψR, k, solρ, solO, Wp))
        @test lhs ≈ conj(rhs) atol = 2e-2 rtol = 2e-2
    end

    @testset "expϕH" begin
        β = 1.1
        dim = (bonddim(ψR), bonddim(ψR))
        solρp, solOp = integrateSol(
            expϕsys!,
            dim,
            (ψL.Q, ψL.R, ψR.Q, ψR.R, β, dim),
            (ψR.Q', ψR.R', ψL.Q', ψL.R', -β, dim),
            ψR.rFP,
            ψL.lFP,
        )
        solρm, solOm = integrateSol(
            expϕsys!,
            dim,
            (ψL.Q, ψL.R, ψR.Q, ψR.R, -β, dim),
            (ψR.Q', ψR.R', ψL.Q', ψL.R', β, dim),
            ψR.rFP,
            ψL.lFP,
        )

        lhs = tr(Wp' * expϕH(ψL, ψR, β, k, solρp, solOp, W))
        rhs = tr(W' * expϕH(ψL, ψR, -β, k, solρm, solOm, Wp))
        @test lhs ≈ conj(rhs) atol = 2e-2 rtol = 2e-2
    end
end
