using LinearAlgebra.LAPACK: geqrf!, ormqr!
using FastLapackInterface
using PolynomialMatrixEquations
#using SolveEyePlusMinusAkronB: EyePlusAtKronBWs, generalized_sylvester_solver!

using LinearAlgebra.BLAS

mutable struct LinearRationalExpectationsWs
    algo::String
    endogenous_nbr::Int64
    exogenous_nbr::Int64
    exogenous_deterministic_nbr::Int64
    forward_indices::Vector{Int64}
    purely_forward_indices::Vector{Int64}
    current_indices::Vector{Int64}
    backward_indices::Vector{Int64}
    both_indices::Vector{Int64}
    static_indices::Vector{Int64}
    dynamic_indices::Vector{Int64}
    non_backward_indices::Vector{Int64}
    current_dynamic_indices::Vector{Int64}
    forward_indices_d::Vector{Int64}
    backward_indices_d::Vector{Int64}
    current_dynamic_indices_d::Vector{Int64}
    current_dynamic_indices_d_j::Vector{Int64}
    exogenous_indices::Vector{Int64}
    static_indices_j::Vector{Int64}
    static_nbr::Int64
    dynamic_nbr::Int64
    forward_nbr::Int64
    backward_nbr::Int64
    both_nbr::Int64
    current_nbr::Int64
    jacobian_static::Matrix{Float64} 
    qr_ws::QRWs
    solver_ws::Union{GsSolverWs, CyclicReductionWs}
    a::Matrix{Float64}
    b::Matrix{Float64}
    c::Matrix{Float64}
    d::Matrix{Float64}
    e::Matrix{Float64}
    x::Matrix{Float64}
    ghx::Matrix{Float64}
    gx::Matrix{Float64}
    hx::Matrix{Float64}
    temp1::Matrix{Float64}
    temp2::Matrix{Float64}
    temp3::Matrix{Float64}
    temp4::Matrix{Float64}
    temp5::Matrix{Float64}
    temp6::Matrix{Float64}
    temp7::Matrix{Float64}
    temp8::Matrix{Float64}
    temp9::Matrix{Float64}
    b10::Matrix{Float64}
    b11::Matrix{Float64}
    icolsD::Vector{Int64}
    icolsE::Vector{Int64}
    jcolsD::Vector{Int64}
    jcolsE::Vector{Int64}
    colsUD::Vector{Int64}
    colsUE::Vector{Int64}
    AGplusB::Matrix{Float64}
    linsolve_static_ws::LUWs
    AGplusB_linsolve_ws::LUWs
    #    eye_plus_at_kron_b_ws::EyePlusAtKronBWs
    
    function LinearRationalExpectationsWs(algo::String,
                                          endogenous_nbr::Int64,
                                          exogenous_nbr::Int64,
                                          exogenous_deterministic_nbr::Int64,
                                          forward_indices::Vector{Int64},
                                          current_indices::Vector{Int64},
                                          backward_indices::Vector{Int64},
                                          both_indices::Vector{Int64},
                                          static_indices::Vector{Int64})
        static_nbr = length(static_indices)
        forward_nbr = length(forward_indices)
        backward_nbr = length(backward_indices)
        both_nbr = length(both_indices)
        current_nbr = length(current_indices)
        dynamic_nbr = endogenous_nbr - static_nbr
        dynamic_indices = setdiff(collect(1:endogenous_nbr), static_indices)
        current_dynamic_indices = setdiff(current_indices, static_indices)
        purely_forward_indices = setdiff(forward_indices, both_indices)
        non_backward_indices = sort(union(purely_forward_indices, static_indices))
        forward_indices_d = findall(in(forward_indices), dynamic_indices)
        backward_indices_d = findall(in(backward_indices), dynamic_indices)
        current_dynamic_indices_d = findall(in(current_dynamic_indices), dynamic_indices)
        current_dynamic_indices_d_j = backward_nbr .+ findall(in(dynamic_indices), current_indices)
        exogenous_indices = backward_nbr + current_nbr + forward_nbr .+ (1:exogenous_nbr)
        jacobian_static = Matrix{Float64}(undef, endogenous_nbr, static_nbr)
        static_indices_j = backward_nbr .+ [findfirst(isequal(x), current_indices) for x in static_indices] 
        qr_ws = QRWs(jacobian_static)
        if algo == "GS"
            de_order = forward_nbr + backward_nbr
            d = zeros(de_order, de_order)
            e = similar(d)
            solver_ws = GsSolverWs(d, backward_nbr)
            a = Matrix{Float64}(undef, 0, 0)
            b = similar(a)
            c = similar(a)
            x = similar(a)
        elseif algo == "CR"
            a = Matrix{Float64}(undef, dynamic_nbr, dynamic_nbr)
            b = similar(a)
            c = similar(a)
            x = similar(a)
            solver_ws = CyclicReductionWs(dynamic_nbr)
            d = Matrix{Float64}(undef, 0, 0)
            e = similar(d)
        end
        ghx = Matrix{Float64}(undef, endogenous_nbr, backward_nbr)
        gx = Matrix{Float64}(undef, forward_nbr, backward_nbr)
        hx = Matrix{Float64}(undef,  backward_nbr, backward_nbr)
        temp1 = Matrix{Float64}(undef, static_nbr, forward_nbr)
        temp2 = Matrix{Float64}(undef, static_nbr, backward_nbr)
        temp3 = Matrix{Float64}(undef, forward_nbr, backward_nbr)
        temp4 = Matrix{Float64}(undef, endogenous_nbr - static_nbr, backward_nbr)
        temp5 = Matrix{Float64}(undef, endogenous_nbr, exogenous_nbr)
        temp6 = Matrix{Float64}(undef, static_nbr, backward_nbr)
        temp7 = Matrix{Float64}(undef, endogenous_nbr, backward_nbr)
        temp8 = Matrix{Float64}(undef, endogenous_nbr, forward_nbr)
        temp9 = Matrix{Float64}(undef, endogenous_nbr, current_nbr)
        b10 = Matrix{Float64}(undef, static_nbr,static_nbr)
        b11 = Matrix{Float64}(undef, static_nbr, endogenous_nbr - static_nbr)
        current_backward_indices = findall(in(backward_indices), current_indices)
        current_forward_indices = findall(in(forward_indices), current_indices)
        # derivatives of current values of variables that are both
        # forward and backward are included in the D matrix
        k1 = findall(in(current_indices), backward_indices)
        icolsD = [k1;  backward_nbr .+ (1:forward_nbr)]
        jcolsD = [backward_nbr .+ current_backward_indices;
                  backward_nbr + current_nbr .+ (1:forward_nbr)]
        k2a = findall(in(purely_forward_indices), forward_indices)
        k2b = findall(in(purely_forward_indices), current_indices)
        icolsE = [1:backward_nbr; backward_nbr .+ k2a]
        jcolsE = [1:backward_nbr; backward_nbr .+ k2b]
        colsUD = findall(in(forward_indices), backward_indices)
        colsUE = backward_nbr .+ findall(in(backward_indices), forward_indices)
        linsolve_static_ws = LUWs(static_nbr)
        AGplusB = Matrix{Float64}(undef, endogenous_nbr, endogenous_nbr)
        AGplusB_linsolve_ws = LUWs(endogenous_nbr)
        #        if m.serially_correlated_exogenous
        #            eye_plus_at_kron_b_ws = EyePlusAtKronBWs(ma, mb, mc, 1)
        #        else
        #            eye_plus_at_kron_b_ws = EyePlusAtKronBWs(1, 1, 1, 1)
        # end
        new(algo, endogenous_nbr, exogenous_nbr,
            exogenous_deterministic_nbr, forward_indices,
            purely_forward_indices, current_indices, backward_indices,
            both_indices, static_indices, dynamic_indices,
            non_backward_indices, current_dynamic_indices,
            forward_indices_d, backward_indices_d,
            current_dynamic_indices_d, current_dynamic_indices_d_j,
            exogenous_indices, static_indices_j, static_nbr,
            dynamic_nbr, forward_nbr, backward_nbr, both_nbr,
            current_nbr, jacobian_static, qr_ws, solver_ws, a, b, c,
            d, e, x, ghx, gx, hx, temp1, temp2, temp3, temp4, temp5,
            temp6, temp7, temp8, temp9, b10, b11, icolsD, icolsE,
            jcolsD, jcolsE, colsUD, colsUE, AGplusB,
            linsolve_static_ws, AGplusB_linsolve_ws)
    end
end

Base.@kwdef struct CyclicReductionOptions
    maxiter::Int64 = 100
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
    g1_1::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    # solution first order derivatives w.r. to current exogenous variables
    g1_2::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    endogenous_variance::Matrix{Float64}
    #    g1_3::SubArray # solution first order derivatives w.r. to lagged exogenous variables
    stationary_variables::Vector{Bool}
    function LinearRationalExpectationsResults(endogenous_nbr::Int64,
                                               exogenous_nbr::Int64,
                                               backward_nbr::Int64)
        state_nbr = backward_nbr + exogenous_nbr
        non_backward_nbr = endogenous_nbr - backward_nbr
        eigenvalues = Vector{Float64}(undef, 0)
        g1 =  zeros(endogenous_nbr,(state_nbr + 1))
        gs1 = zeros(backward_nbr,backward_nbr)
        hs1 = zeros(backward_nbr,exogenous_nbr)
        gns1 = zeros(non_backward_nbr,backward_nbr)
        hns1 = zeros(non_backward_nbr,exogenous_nbr)
        g1_1 = view(g1, :, 1:backward_nbr)
        g1_2 = view(g1, :, backward_nbr .+ (1:exogenous_nbr))
        endogenous_variance = zeros(endogenous_nbr, endogenous_nbr)
        stationary_variables = Vector{Bool}(undef, endogenous_nbr)
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
p_static: a vector of indices of static variables in jacobian matrix
ws: FirstOrderWs workspace. On exit contains the triangular part conrresponding
                                    to static variables in jacobian_static
"""
function remove_static!(jacobian::Matrix{Float64},
                        ws::LinearRationalExpectationsWs)
    ws.jacobian_static .= view(jacobian, :, ws.static_indices_j)
    geqrf!(ws.qr_ws, ws.jacobian_static)
    ormqr!(ws.qr_ws, 'L', 'T', ws.jacobian_static, jacobian)
end

"""
Computes the solution for the static variables:
G_y,static = -B_s,s^{-1}(A_s*Gy,fwrd*Gs + B_s,d*Gy,dynamic + C_s) 
""" 
function add_static!(results::LinearRationalExpectationsResults,
                     jacobian::Matrix{Float64},
                     ws::LinearRationalExpectationsWs)
    @views @inbounds begin
        # static rows are at the top of the QR transformed Jacobian matrix
        # B_s,s
        # fill!(ws.b10, 0.0)
        stat_r = 1:ws.static_nbr
        back_r = 1:ws.backward_nbr
        # B_s,d
        ws.b10 .= jacobian[stat_r, ws.static_indices_j]
        ws.b11[:, ws.current_dynamic_indices_d] .= jacobian[stat_r, ws.current_dynamic_indices_d_j]
        # A_s
        ws.temp1 .= jacobian[stat_r, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr)]
        # C_s
        ws.temp2 .= jacobian[stat_r, back_r]
        # Gy.fwrd
        ws.temp3 .= results.g1_1[ws.forward_indices, back_r]
        # Gy.dynamic
        ws.temp4 .= results.g1_1[ws.dynamic_indices, back_r]
        # ws.temp2 = B_s,d*Gy.dynamic + C_s
        mul!(ws.temp2, ws.b11, ws.temp4, 1.0, 1.0)
        # ws.temp6 = A_s*Gy.fwrd*gs1
        mul!(ws.temp6, ws.temp1, ws.temp3)
        mul!(ws.temp2, ws.temp6, results.gs1, -1.0, -1.0)
        # ws.temp3 = B_s,s\ws.temp2
        
        lu_t = LU(factorize!(ws.linsolve_static_ws, ws.b10)...)
        ldiv!(lu_t, ws.temp2)
        
        results.g1[ws.static_indices, back_r] .= ws.temp2
    end
    return results.g1, jacobian
end

function get_abc!(ws::LinearRationalExpectationsWs,
                  jacobian::AbstractMatrix{Float64})
    fill!(ws.a, 0.0)
    fill!(ws.b, 0.0)
    fill!(ws.c, 0.0)

    dyn_r = ws.static_nbr .+ (1:ws.dynamic_nbr)
    @views @inbounds begin
        ws.a[:, ws.forward_indices_d]         .= jacobian[dyn_r, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr)]
        ws.b[:, ws.current_dynamic_indices_d] .= jacobian[dyn_r, ws.current_dynamic_indices_d_j]
        ws.c[:, ws.backward_indices_d]        .= jacobian[dyn_r, 1:ws.backward_nbr]
    end
    return ws.a, ws.b, ws.c
end

function get_de!(ws::LinearRationalExpectationsWs,
                 jacobian::AbstractMatrix{Float64})
    fill!(ws.d, 0.0)
    fill!(ws.e, 0.0)
    r = ws.static_nbr .+ (1:ws.dynamic_nbr)
    dyn_r  = 1:ws.dynamic_nbr
    both_r = 1:ws.both_nbr
    @views @inbounds begin
        ws.d[dyn_r, ws.icolsD] .=  jacobian[r, ws.jcolsD]
        ws.e[dyn_r, ws.icolsE] .=  .- jacobian[r, ws.jcolsE]
        for i = both_r 
            k = ws.dynamic_nbr + i
            ws.d[k, ws.colsUD[i]] = 1.0
            ws.e[k, ws.colsUE[i]] = 1.0
        end
    end
    return ws.d, ws.e
end

function make_AGplusB!(AGplusB::AbstractMatrix{Float64},
                          A::AbstractMatrix{Float64},
                          G::AbstractMatrix{Float64},
                          B::AbstractMatrix{Float64},
                          ws::LinearRationalExpectationsWs)
    fill!(AGplusB, 0.0)
    @views @inbounds begin
        AGplusB[:, ws.current_indices] .= B
        ws.temp3 .= G[ws.forward_indices, :]
        mul!(ws.temp7, A, ws.temp3)
        AGplusB[:, ws.backward_indices] .+= ws.temp7
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
    if ws.exogenous_nbr > 0
        results.g1_2 .= .-view(jacobian, :, ws.exogenous_indices)
        
#        if ws.serially_correlated_exogenous
            # TO BE DONE
        #        else
        lu_t = LU(factorize!(ws.AGplusB_linsolve_ws, ws.AGplusB)...)
        ldiv!(lu_t, results.g1_2)
#        end
    end
end
     
function first_order_solver!(results::LinearRationalExpectationsResults,
                             algo::String,
                             jacobian::AbstractMatrix{Float64},
                             options::LinearRationalExpectationsOptions,
                             ws::LinearRationalExpectationsWs)
    remove_static!(jacobian, ws)
    back_r     = 1:ws.backward_nbr
    back_ids   = ws.backward_indices
    back_ids_d = ws.backward_indices_d
    dyn_ids    = ws.dynamic_indices
    pur_for_ids = ws.purely_forward_indices
    @views @inbounds begin
        if algo == "CR"
            ws.a, ws.b, ws.c = get_abc!(ws, jacobian)
            cyclic_reduction!(ws.x, ws.c, ws.b, ws.a, ws.solver_ws,
                              options.cyclic_reduction.tol, options.cyclic_reduction.maxiter)
            results.gs1[:, back_r] .= ws.x[back_ids_d, back_ids_d]
            results.g1[dyn_ids, back_r] .= ws.x[:, back_ids_d]
        elseif algo == "GS"
            ws.d, ws.e = get_de!(ws, jacobian)
            try
                gs_solver!(ws.solver_ws, ws.d, ws.e, ws.backward_nbr,
                           options.generalized_schur.criterium)
            catch e
                resize!(results.eigenvalues, length(ws.solver_ws.schurws.eigen_values))
                copy!(results.eigenvalues, ws.solver_ws.eigen_values)
                rethrow(e)
            end

            results.gs1 .= ws.solver_ws.g1
            results.g1[back_ids, back_r] .= ws.solver_ws.g1[back_r, back_r]
            results.g1[pur_for_ids, back_r] .= ws.solver_ws.g2[ws.icolsE[ws.backward_nbr .+ (1:(ws.forward_nbr - ws.both_nbr))] .- ws.backward_nbr, :]
            resize!(results.eigenvalues, length(ws.solver_ws.schurws.eigen_values))
            results.eigenvalues .= ws.solver_ws.schurws.eigen_values
        else
            error("Algorithm $algo not recognized")
        end
        if ws.static_nbr > 0
            results.g1, jacobian = add_static!(results, jacobian, ws)
        end
        #    A = view(jacobian, :, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
        #    B = view(jacobian, :, ws.backward_nbr .+ ws.current_indices)
        ws.temp8[:, 1:ws.forward_nbr] .= jacobian[:, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr)]
        ws.temp9[:, 1:ws.current_nbr] .= jacobian[:, ws.backward_nbr .+ (1:ws.current_nbr)]
        ws.AGplusB = make_AGplusB!(ws.AGplusB, ws.temp8, results.g1_1, ws.temp9, ws)        
        solve_for_derivatives_with_respect_to_shocks!(results, jacobian, ws)
        results.hs1 .= results.g1_2[back_ids, :]
        results.gns1 .= results.g1_1[ws.non_backward_indices, :]
        results.hns1 .= results.g1_2[ws.non_backward_indices, :]
    end
end


