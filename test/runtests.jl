using Test
using RCMPS

@testset "RCMPS.jl" begin
    include("chi1.jl")
    include("hermiticity.jl")
end
