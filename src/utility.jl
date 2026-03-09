J(x) = besselk(0.25, abs(x)) / (2^0.25 * sqrt(pi) * gamma(0.25) * abs(x)^0.25)

CC(x, y) = x * y - y * x

function sysJ(sys!)
    function sysJ!(Jv, v, ρ, p, x)
        dρ = reshape(Jv, size(ρ))
        ρv = reshape(v, size(ρ))
        sys!(dρ, ρv, p, x, false)
        nothing
    end
    return sysJ!
end

# function integrateODE(F!, u0, p, fullsol=false; integration_limit=integration_limit, ode_tol=ode_tol)
#     J! = sysJ(F!)
#     prob = DE.ODEProblem(DE.ODEFunction(F!, jvp=J!), u0, (-integration_limit, integration_limit), p)

#     if fullsol
#         sol = DE.solve(prob, DE.Rodas5Pe(autodiff=false, linsolve=DE.KrylovJL_GMRES()), abstol=ode_tol, reltol=ode_tol)
#         return sol
#     else
#         sol = DE.solve(prob, DE.Rodas5Pe(autodiff=false, linsolve=DE.KrylovJL_GMRES()), abstol=ode_tol, reltol=ode_tol, save_on=false)
#         return sol.u[end]
#     end
# end

function integrateODE(F!, u0, p, fullsol=false; integration_limit=integration_limit, ode_tol=ode_tol)
    tlim = cbrt(integration_limit)

    prob = DE.ODEProblem(u0, (-tlim, tlim), p) do du, u, p, t
        F!(du, u, p, t^3)
        du .*= 3 * t^2
    end

    sol = DE.solve(prob, DE.Vern7(), abstol=ode_tol, reltol=ode_tol, save_on=fullsol)
    if fullsol
        return (x -> sol(cbrt(x)))
    else
        return sol.u[end]
    end
end

function integrateSol(sys!, dim, pρ, pO, ρ0=zeros(ComplexF64, dim), O0=zeros(ComplexF64, dim))
    solρ = integrateODE(sys!, vec(ρ0), pρ, true)
    solO = integrateODE(sys!, vec(O0), pO, true)
    return solρ, solO
end

function gdlinesearch(f, fg, x0, inner, retract; c=1e-3, α0=1., fac=0.5, maxiter=100, gradtol=1e-3, verbose=false, update_cycle=6)
    x = x0
    fval = 0.0

    iter_ave = 1.
    p = 0.3
    for i in 1:maxiter
        fval, g = fg(x)
        ss = sqrt(max(inner(x, g, g), 1e-16))
        if verbose
            println("Iter $i: f = $fval, |grad| = $ss")
        end
        if ss < gradtol
            return fval, x
        end

        α = α0
        xp, _ = retract(x, g, -α)
        for i = 1:20
            fp = f(xp)
            if verbose
                println("  Line search iter $i: α = $α, f = $fp")
            end
            if fp < fval - c * α * ss^2
                iter_ave = iter_ave * (1 - p) + i * p
                break
            end
            α *= fac
            xp, _ = retract(x, g, -α)
        end

        if iter_ave < 1.5
            α0 /= fac
        elseif iter_ave > 1.9
            α0 *= fac
        end
        if verbose
            println("  - Rolling average iter $(round(iter_ave, digits=1)), updating α0 to $α0")
        end

        x = xp
    end
    return fval, x
end
