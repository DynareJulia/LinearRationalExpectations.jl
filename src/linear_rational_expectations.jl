using LinearAlgebra.LAPACK: geqrf!, ormqr!
using FastLapackInterface
using PolynomialMatrixEquations
#using SolveEyePlusMinusAkronB: EyePlusAtKronBWs, generalized_sylvester_solver!

using LinearAlgebra.BLAS

struct Indices
    current        ::Vector{Int}
    forward        ::Vector{Int}
    purely_forward ::Vector{Int}
    backward       ::Vector{Int}
    both           ::Vector{Int}
    non_backward   ::Vector{Int} #union(purely_forward, static)
    
    static          ::Vector{Int}
    dynamic         ::Vector{Int}
    dynamic_current ::Vector{Int}
    
    current_in_dynamic ::Vector{Int} #ids of dynamic that are current
    forward_in_dynamic ::Vector{Int} #ids of dynamic that are forward
    backward_in_dynamic::Vector{Int} #ids of dynamic that are backward

    current_in_dynamic_jacobian ::Vector{Int}
    current_in_static_jacobian  ::Vector{Int}
    
    exogenous   ::Vector{Int}
    n_endogenous ::Int

    D_columns::NamedTuple{(:D, :jacobian), NTuple{2, Vector{Int}}}
    E_columns::NamedTuple{(:E, :jacobian), NTuple{2, Vector{Int}}}
    UD_columns::Vector{Int}
    UE_columns::Vector{Int}
    
end

function Indices(n_exogenous::Int, forward::Vector{Int}, current::Vector{Int}, backward::Vector{Int}, static::Vector{Int})
    n_forward  = length(forward)
    n_backward = length(backward)
    n_current  = length(current)
    
    n_endogenous = maximum(Iterators.flatten((forward, backward, current)))
    exogenous = n_backward + n_current + n_forward .+ (1:n_exogenous)
    
    both            = intersect(forward, backward)
    dynamic         = setdiff(collect(1:n_endogenous), static)
    current_dynamic = setdiff(current, static)
    purely_forward  = setdiff(forward, both)
    non_backward    = sort(union(purely_forward, static))
    
    forward_in_dynamic          = findall(in(forward), dynamic)
    backward_in_dynamic         = findall(in(backward), dynamic)
    current_in_dynamic          = findall(in(current_dynamic), dynamic)
    current_in_dynamic_jacobian = n_backward .+ findall(in(dynamic), current)
    current_in_static_jacobian  = n_backward .+ [findfirst(isequal(x), current) for x in static]

    # derivatives of current values of variables that are both
    # forward and backward are included in the D matrix
    k1 = findall(in(current), backward)
    k2a = findall(in(purely_forward), forward)
    k2b = findall(in(purely_forward), current)

    D_columns = (D        = [k1; n_backward .+ (1:n_forward)],
                 jacobian = [n_backward .+ findall(in(backward), current);
                             n_backward + n_current .+ (1:n_forward)])

    E_columns = (E        = [1:n_backward; n_backward .+ k2a],
                 jacobian = [1:n_backward; n_backward .+ k2b])
    
    UD_columns = findall(in(forward), backward)
    UE_columns = n_backward .+ findall(in(backward), forward)
    
    return Indices(
        current,
        forward,
        purely_forward,
        backward,
        both,
        non_backward,

        static,
        dynamic,
        current_dynamic,

        current_in_dynamic,
        forward_in_dynamic,
        backward_in_dynamic,
        current_in_dynamic_jacobian,
        current_in_static_jacobian,
        
        exogenous,
        n_endogenous,

        D_columns,
        E_columns,
        UD_columns,
        UE_columns,
    )
end

n_static(i::Indices)     = length(i.static)
n_forward(i::Indices)    = length(i.forward)
n_backward(i::Indices)   = length(i.backward)
n_both(i::Indices)       = length(i.both)
n_current(i::Indices)    = length(i.current)
n_dynamic(i::Indices)    = length(i.dynamic)
n_endogenous(i::Indices) = i.n_endogenous
n_exogenous(i::Indices) = length(i.exogenous)

mutable struct LinearGsSolverWs
    solver_ws::GsSolverWs
    ids::Indices
    d::Matrix{Float64}
    e::Matrix{Float64}
end
function LinearGsSolverWs(ids::Indices)
    n_back = n_backward(ids)
    de_order = n_forward(ids) + n_back
    d = zeros(de_order, de_order)
    e = similar(d)
    solver_ws = GsSolverWs(d, n_back)
    return LinearGsSolverWs(solver_ws, ids, d, e)
end

function copy_jacobian!(ws::LinearGsSolverWs,
                        jacobian::AbstractMatrix{Float64})
    fill!(ws.d, 0.0)
    fill!(ws.e, 0.0)

    ids = ws.ids
    
    n_dyn  = n_dynamic(ids)
    dyn_r  = 1:n_dyn
    r      = n_static(ids) .+ dyn_r
    both_r = 1:n_both(ids)
    @views @inbounds begin
        ws.d[dyn_r, ids.D_columns.D] .=     jacobian[r, ids.D_columns.jacobian]
        ws.e[dyn_r, ids.E_columns.E] .=  .- jacobian[r, ids.E_columns.jacobian]
        for i = both_r 
            k = n_dyn + i
            ws.d[k, ids.UD_columns[i]] = 1.0
            ws.e[k, ids.UE_columns[i]] = 1.0
        end
    end
    return ws.d, ws.e
end

solve!(ws::LinearGsSolverWs, crit) = gs_solver!(ws.solver_ws, ws.d, ws.e, n_backward(ws.ids), crit)

function solve_g1!(results, ws::LinearGsSolverWs, options)
    ids = ws.ids
    back    = ids.backward
    n_back  = n_backward(ids)
    back_r  = 1:n_back
    pur_for = ids.purely_forward
    try
        solve!(ws, options.generalized_schur.criterium)
    catch e
        resize!(results.eigenvalues, length(ws.solver_ws.schurws.eigen_values))
        copy!(results.eigenvalues, ws.solver_ws.schurws.eigen_values)
        rethrow(e)
    end

    results.gs1                 .= ws.solver_ws.g1
    results.g1[back, back_r]    .= ws.solver_ws.g1[back_r, back_r]
    results.g1[pur_for, back_r] .= ws.solver_ws.g2[ids.E_columns.E[n_back .+ (1:(n_forward(ids) - n_both(ids)))] .- n_back, :]

    resize!(results.eigenvalues, length(ws.solver_ws.schurws.eigen_values))
    results.eigenvalues .= ws.solver_ws.schurws.eigen_values
    return results.g1, results.gs1
end

mutable struct LinearCyclicReductionWs
    solver_ws::CyclicReductionWs
    ids::Indices
    a::Matrix{Float64}
    b::Matrix{Float64}
    c::Matrix{Float64}
    x::Matrix{Float64}
end
function LinearCyclicReductionWs(ids::Indices)
    n_dyn  = n_dynamic(ids)
    a = Matrix{Float64}(undef, n_dyn, n_dyn)
    b = similar(a)
    c = similar(a)
    x = similar(a)
    solver_ws = CyclicReductionWs(n_dyn)
    return LinearCyclicReductionWs(solver_ws, ids, a, b, c, x)
end

function copy_jacobian!(ws::LinearCyclicReductionWs,
                        jacobian::AbstractMatrix{Float64})
    fill!(ws.a, 0.0)
    fill!(ws.b, 0.0)
    fill!(ws.c, 0.0)
    ids = ws.ids
    dyn_r = n_static(ids) .+ (1:n_dynamic(ids))
    @views @inbounds begin
        ws.a[:, ids.forward_in_dynamic]  .= jacobian[dyn_r, n_backward(ids) + n_current(ids) .+ (1:n_forward(ids))]
        ws.b[:, ids.current_in_dynamic]  .= jacobian[dyn_r, ids.current_in_dynamic_jacobian]
        ws.c[:, ids.backward_in_dynamic] .= jacobian[dyn_r, 1:n_backward(ids)]
    end
    return ws.a, ws.b, ws.c
end

solve!(ws::LinearCyclicReductionWs, tol, maxiter) = cyclic_reduction!(ws.x, ws.c, ws.b, ws.a, ws.solver_ws, tol, maxiter)

function solve_g1!(results, ws::LinearCyclicReductionWs, options)
    ids = ws.ids
    back_r = 1:n_backward(ids)
    dyn = ids.dynamic
    back_d  = ids.backward_in_dynamic
    solve!(ws, options.cyclic_reduction.tol, options.cyclic_reduction.maxiter)
                      
    results.gs1[:, back_r]  .= ws.x[back_d, back_d]
    results.g1[dyn, back_r] .= ws.x[:, back_d]
    return results.g1, results.gs1
end

mutable struct LinearRationalExpectationsWs
    ids::Indices
    jacobian_static::Matrix{Float64} 
    qr_ws::QRWs
    ormqr_ws::QROrmWs
   
    solver_ws::Union{LinearGsSolverWs, LinearCyclicReductionWs}
    A_s::Matrix{Float64}
    C_s::Matrix{Float64}
    Gy_forward::Matrix{Float64}
    Gy_dynamic::Matrix{Float64}
    temp::Matrix{Float64}
    AGplusB_backward::Matrix{Float64}
    jacobian_forward::Matrix{Float64}
    jacobian_current::Matrix{Float64}
    b10::Matrix{Float64}
    b11::Matrix{Float64}
    AGplusB::Matrix{Float64}
    linsolve_static_ws::LUWs
    AGplusB_linsolve_ws::LUWs
    #    eye_plus_at_kron_b_ws::EyePlusAtKronBWs
    
end
function LinearRationalExpectationsWs(solver_ws::Union{LinearGsSolverWs, LinearCyclicReductionWs}, ids::Indices)
    n_back = n_backward(ids)
    n_stat = n_static(ids)
    n_forw = n_forward(ids)
    n_end  = n_endogenous(ids)
    n_exo  = n_exogenous(ids)
    n_curr = n_current(ids)
    n = n_back + n_curr + n_forw + n_exo
    
    jacobian_static = Matrix{Float64}(undef, n_endogenous(ids), n_static(ids))
     
    qr_ws = QRWs(jacobian_static)
    ormqr_ws = QROrmWs(qr_ws, 'L', 'T', jacobian_static, zeros(n_end, n))
    
    A_s = Matrix{Float64}(undef, n_stat, n_forw)
    C_s = Matrix{Float64}(undef, n_stat, n_back)
    Gy_forward = Matrix{Float64}(undef, n_forw, n_back)
    
    Gy_dynamic = Matrix{Float64}(undef, n_end - n_stat, n_back)
    temp = Matrix{Float64}(undef, n_stat, n_back)

    AGplusB_backward = Matrix{Float64}(undef, n_end, n_back)

    jacobian_forward = Matrix{Float64}(undef, n_end, n_forw)
    jacobian_current = Matrix{Float64}(undef, n_end, n_curr)

    b10 = Matrix{Float64}(undef, n_stat,n_stat)
    b11 = Matrix{Float64}(undef, n_stat, n_end - n_stat)
    linsolve_static_ws = LUWs(n_stat)
    AGplusB = Matrix{Float64}(undef, n_end, n_end)
    AGplusB_linsolve_ws = LUWs(n_end)
    LinearRationalExpectationsWs(ids, jacobian_static, qr_ws, ormqr_ws, solver_ws, A_s,
                                 C_s, Gy_forward, Gy_dynamic, 
                                 temp, AGplusB_backward, jacobian_forward, jacobian_current,
                                 b10, b11, AGplusB,
                                 linsolve_static_ws, AGplusB_linsolve_ws)
end

function LinearRationalExpectationsWs(algo::String, ids::Indices)
    solver_ws = algo == "GS" ? LinearGsSolverWs(ids) : LinearCyclicReductionWs(ids)
    return LinearRationalExpectationsWs(solver_ws, ids)
end

function LinearRationalExpectationsWs(algo::String, args...)
    ids = Indices(args...)
    solver_ws = algo == "GS" ? LinearGsSolverWs(ids) : LinearCyclicReductionWs(ids)
    return LinearRationalExpectationsWs(solver_ws, ids)
end

Base.@kwdef struct CyclicReductionOptions
    maxiter::Int = 100
    tol::Float64   = 1e-8
end

Base.@kwdef struct GeneralizedSchurOptions
    # Near unit roots are considered stable roots
    criterium::Float64 = 1.0 + 1e-6
end

Base.@kwdef struct LinearRationalExpectationsOptions
    cyclic_reduction::CyclicReductionOptions = CyclicReductionOptions()
    generalized_schur::GeneralizedSchurOptions = GeneralizedSchurOptions()
end

mutable struct LinearRationalExpectationsResults
    eigenvalues::Vector{Complex{Float64}}
    g1::Matrix{Float64}  # full approximation
    gs1::Matrix{Float64} # state transition matrices: states x states
    hs1::Matrix{Float64} # states x shocks
    gns1::Matrix{Float64} # non states x states
    hns1::Matrix{Float64} # non states x shocsks
    # solution first order derivatives w.r. to state variables
    g1_1::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    # solution first order derivatives w.r. to current exogenous variables
    g1_2::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    endogenous_variance::Matrix{Float64}
    #    g1_3::SubArray # solution first order derivatives w.r. to lagged exogenous variables
    stationary_variables::Vector{Bool}
    function LinearRationalExpectationsResults(n_endogenous::Int,
                                               exogenous_nbr::Int,
                                               backward_nbr::Int)
        state_nbr = backward_nbr + exogenous_nbr
        non_backward_nbr = n_endogenous - backward_nbr
        eigenvalues = Vector{Float64}(undef, 0)
        g1 =  zeros(n_endogenous,(state_nbr + 1))
        gs1 = zeros(backward_nbr,backward_nbr)
        hs1 = zeros(backward_nbr,exogenous_nbr)
        gns1 = zeros(non_backward_nbr,backward_nbr)
        hns1 = zeros(non_backward_nbr,exogenous_nbr)
        g1_1 = view(g1, :, 1:backward_nbr)
        g1_2 = view(g1, :, backward_nbr .+ (1:exogenous_nbr))
        endogenous_variance = zeros(n_endogenous, n_endogenous)
        stationary_variables = Vector{Bool}(undef, n_endogenous)
        new(eigenvalues, g1, gs1, hs1, gns1, hns1, g1_1, g1_2, endogenous_variance, stationary_variables)
#        g1_3 = view(g[1], :, backward_nbr + exogenous_nbr .+ lagged_exogenous_nbr)
#        new(g, gs, g1_1, g1_2, g1_3, AGplusB, AGplusB_linsolve_ws)
    end
end

"""
remove_static! removes a subset of variables (columns) and rows by QR decomposition
jacobian: on entry jacobian matrix of the original model
          on exit transformed jacobian. The rows corresponding to the dynamic part 
                  are at the bottom
p_static: a vector of ids of static variables in jacobian matrix
ws: FirstOrderWs workspace. On exit contains the triangular part conrresponding
                                    to static variables in jacobian_static
"""
function remove_static!(jacobian::Matrix{Float64},
                        ws::LinearRationalExpectationsWs)
    ws.jacobian_static .= view(jacobian, :, ws.ids.current_in_static_jacobian)
    geqrf!(ws.qr_ws, ws.jacobian_static)
    ormqr!(ws.ormqr_ws, 'L', 'T', ws.jacobian_static, jacobian)
end

"""
Computes the solution for the static variables:
G_y,static = -B_s,s^{-1}(A_s*Gy,fwrd*Gs + B_s,d*Gy,dynamic + C_s) 
""" 
function add_static!(results::LinearRationalExpectationsResults,
                     jacobian::Matrix{Float64},
                     ws::LinearRationalExpectationsWs)
    ids = ws.ids
    @views @inbounds begin
        # static rows are at the top of the QR transformed Jacobian matrix
        stat_r = 1:n_static(ids)
        back_r = 1:n_backward(ids)
        # B_s,s
        # fill!(ws.b10, 0.0)
        ws.b10 .= jacobian[stat_r, ids.current_in_static_jacobian]
        # B_s,d
        ws.b11[:, ids.current_in_dynamic] .= jacobian[stat_r, ids.current_in_dynamic_jacobian]
        # A_s
        ws.A_s .= jacobian[stat_r, n_backward(ids) + n_current(ids) .+ (1:n_forward(ids))]
        # C_s
        ws.C_s .= jacobian[stat_r, back_r]
        # Gy.fwrd
        ws.Gy_forward .= results.g1_1[ids.forward, back_r]
        # Gy.dynamic
        ws.Gy_dynamic .= results.g1_1[ids.dynamic, back_r]
        # ws.C_s = B_s,d*Gy.dynamic + C_s
        
        mul!(ws.C_s, ws.b11, ws.Gy_dynamic, 1.0, 1.0)
        # ws.temp = A_s*Gy.fwrd*gs1
        mul!(ws.temp, ws.A_s, ws.Gy_forward)
        mul!(ws.C_s, ws.temp, results.gs1, -1.0, -1.0)
        # ws.Gy_forward = B_s,s\ws.C_s
        
        lu_t = LU(factorize!(ws.linsolve_static_ws, ws.b10)...)
        ldiv!(lu_t, ws.C_s)
        
        results.g1[ids.static, back_r] .= ws.C_s
    end
    return results.g1, jacobian
end

function make_AGplusB!(AGplusB::AbstractMatrix{Float64},
                          A::AbstractMatrix{Float64},
                          G::AbstractMatrix{Float64},
                          B::AbstractMatrix{Float64},
                          ws::LinearRationalExpectationsWs)
    fill!(AGplusB, 0.0)
    ids = ws.ids
    @views @inbounds begin
        AGplusB[:, ids.current] .= B
        ws.Gy_forward .= G[ids.forward, :]
        mul!(ws.AGplusB_backward, A, ws.Gy_forward)
        AGplusB[:, ids.backward] .+= ws.AGplusB_backward
        return AGplusB
    end
    
end

function solve_for_derivatives_with_respect_to_shocks!(results::LinearRationalExpectationsResults,
                                                       jacobian::AbstractMatrix{Float64},
                                                       ws::LinearRationalExpectationsWs)
    #=
    if model.lagged_exogenous_nbr > 0
        f6 = view(jacobian,:,model.i_lagged_exogenous)
        for i = 1:model.current_exogenous_nbr
            for j = 1:model.endo_nbr
                results.g1_3[i,j] = -f6[i,j]
            end
        end
        linsolve_core_no_lu!(results.f1g1plusf2, results.g1_3, ws)
    end
    =#
    if n_exogenous(ws.ids) > 0
        results.g1_2 .= .-view(jacobian, :, ws.ids.exogenous)
        
#        if ws.serially_correlated_exogenous
            # TO BE DONE
        #        else
        lu_t = LU(factorize!(ws.AGplusB_linsolve_ws, ws.AGplusB)...)
        ldiv!(lu_t, results.g1_2)
#        end
    end
end
     
function first_order_solver!(results::LinearRationalExpectationsResults,
                             jacobian::AbstractMatrix{Float64},
                             options::LinearRationalExpectationsOptions,
                             ws::LinearRationalExpectationsWs)
                             
    ids = ws.ids
    if n_static(ids) > 0
        remove_static!(jacobian, ws)
    end
    
    n_back    = n_backward(ids)
    n_cur     = n_current(ids)
    forward_r = 1:n_forward(ids)
    current_r = 1:n_cur
    back      = ids.backward
    
    @views @inbounds begin
        copy_jacobian!(ws.solver_ws, jacobian)
        results.g1, results.gs1 = solve_g1!(results, ws.solver_ws, options)
        if n_static(ids) > 0
            results.g1, jacobian = add_static!(results, jacobian, ws)
        end
        #    A = view(jacobian, :, n_backward(ws.ids) + ws.current_nbr .+ (1:ws.forward_nbr))
        #    B = view(jacobian, :, n_backward(ws.ids) .+ ws.ids.current)
        ws.jacobian_forward[:, forward_r] .= jacobian[:, n_back + n_cur .+ forward_r]
        ws.jacobian_current[:, current_r] .= jacobian[:, n_back .+ current_r]
        
        ws.AGplusB = make_AGplusB!(ws.AGplusB, ws.jacobian_forward, results.g1_1, ws.jacobian_current, ws)        
        solve_for_derivatives_with_respect_to_shocks!(results, jacobian, ws)
        
        results.hs1  .= results.g1_2[back, :]
        results.gns1 .= results.g1_1[ids.non_backward, :]
        results.hns1 .= results.g1_2[ids.non_backward, :]
    end
end
