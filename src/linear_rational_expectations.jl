using LinearAlgebra
using FastLapackInterface
using FastLapackInterface.LinSolveAlgo
using FastLapackInterface.QrAlgo
using PolynomialMatrixEquations
#using SolveEyePlusMinusAkronB: EyePlusAtKronBWs, generalized_sylvester_solver!

using LinearAlgebra.BLAS

struct LinearRationalExpectationsWs
    algo::String
    endogenous_nbr
    exogenous_nbr
    exogenous_deterministic_nbr
    forward_indices
    purely_forward_indices
    current_indices
    backward_indices
    both_indices
    static_indices
    dynamic_indices
    current_dynamic_indices
    forward_indices_d
    backward_indices_d
    current_dynamic_indices_d
    exogenous_indices
    static_nbr
    forward_nbr
    backward_nbr
    both_nbr
    current_nbr
    jacobian_static::Matrix{Float64} 
    qr_ws::QrWs
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
    b10::Matrix{Float64}
    b11::Matrix{Float64}
    icolsD::Vector{Int64}
    icolsE::Vector{Int64}
    jcolsD::Vector{Int64}
    jcolsE::Vector{Int64}
    colsUD::Vector{Int64}
    colsUE::Vector{Int64}
    AGplusB::Matrix{Float64}
    linsolve_static_ws::LinSolveWs
    AGplusB_linsolve_ws::LinSolveWs
    #    eye_plus_at_kron_b_ws::EyePlusAtKronBWs
    
    function LinearRationalExpectationsWs(algo::String,
                                          endogenous_nbr,
                                          exogenous_nbr,
                                          exogenous_deterministic_nbr,
                                          forward_indices,
                                          current_indices,
                                          backward_indices,
                                          both_indices,
                                          static_indices)
        static_nbr = length(static_indices)
        forward_nbr = length(forward_indices)
        backward_nbr = length(backward_indices)
        both_nbr = length(both_indices)
        current_nbr = length(current_indices)
        dynamic_nbr = endogenous_nbr - static_nbr
        dynamic_indices = setdiff(collect(1:endogenous_nbr), static_indices)
        current_dynamic_indices = setdiff(current_indices, static_indices)
        purely_forward_indices = setdiff(forward_indices, both_indices)
        forward_indices_d = findall(in(forward_indices), dynamic_indices)
        backward_indices_d = findall(in(backward_indices), dynamic_indices)
        current_dynamic_indices_d = findall(in(current_dynamic_indices), dynamic_indices)
        exogenous_indices = backward_nbr + current_nbr + forward_nbr .+ (1:exogenous_nbr)
        if static_nbr > 0
            jacobian_static = Matrix{Float64}(undef, endogenous_nbr, static_nbr)
            qr_ws = QrWs(jacobian_static)
#        else
#            jacobian_static = Matrix{Float64}(undef, 0,0)
#            qr_ws = QrWs(Matrix{Float64}(undef, 0,0))
        end
        if algo == "GS"
            de_order = forward_nbr + backward_nbr
            d = zeros(de_order, de_order)
            e = similar(d)
            solver_ws = GsSolverWs(d, e, backward_nbr)
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
        k2 = findall(in(current_indices), forward_indices)
        k2a = findall(in(purely_forward_indices), forward_indices[k2])
        icolsE = [1:backward_nbr; backward_nbr .+ k2a]
        jcolsE = [1:backward_nbr; backward_nbr .+ forward_indices[k2a]]
        colsUD = findall(in(forward_indices), backward_indices)
        colsUE = backward_nbr .+ findall(in(backward_indices), forward_indices)
        linsolve_static_ws = LinSolveWs(static_nbr)
        AGplusB = Matrix{Float64}(undef, endogenous_nbr, endogenous_nbr)
        AGplusB_linsolve_ws = LinSolveAlgo.LinSolveWs(endogenous_nbr)
        #        if m.serially_correlated_exogenous
        #            eye_plus_at_kron_b_ws = EyePlusAtKronBWs(ma, mb, mc, 1)
        #        else
        #            eye_plus_at_kron_b_ws = EyePlusAtKronBWs(1, 1, 1, 1)
        # end
        new(algo, endogenous_nbr, exogenous_nbr, exogenous_deterministic_nbr,
            forward_indices, purely_forward_indices, current_indices,
            backward_indices, both_indices, static_indices, dynamic_indices,
            current_dynamic_indices, forward_indices_d, backward_indices_d,
            current_dynamic_indices_d, exogenous_indices, static_nbr, forward_nbr, backward_nbr,
            both_nbr, current_nbr, jacobian_static, qr_ws, solver_ws,
            a, b, c, d, e, x,
            ghx, gx, hx, temp1, temp2, temp3, temp4, temp5, temp6, temp7, b10, b11,
            icolsD, icolsE, jcolsD, jcolsE,
            colsUD, colsUE, AGplusB, linsolve_static_ws, AGplusB_linsolve_ws)
    end
end

struct LinearRationalExpectationsResults
    g1::Matrix{Float64}  # full approximation
    gs1::Matrix{Float64} # state transition matrices
    g1_1::SubArray # solution first order derivatives w.r. to state variables
    g1_2::SubArray # solution first order derivatives w.r. to current exogenous variables
#    g1_3::SubArray # solution first order derivatives w.r. to lagged exogenous variables
    
    function LinearRationalExpectationsResults(order::Int64, endogenous_nbr::Int64, exogenous_nbr::Int64, backward_nbr::Int64)
        nstate = backward_nbr + exogenous_nbr 
        g1 =  zeros(endogenous_nbr,(nstate + 1))
        gs1 = zeros(backward_nbr,backward_nbr)
        g1_1 = view(g1, :, 1:backward_nbr)
        g1_2 = view(g1, :, backward_nbr .+ (1:exogenous_nbr))
#        g1_3 = view(g[1], :, backward_nbr + exogenous_nbr .+ lagged_exogenous_nbr)
#        new(g, gs, g1_1, g1_2, g1_3, AGplusB, AGplusB_linsolve_ws)
        new(g1, gs1, g1_1, g1_2)
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
function remove_static!(jacobian::Matrix,
                        ws::LinearRationalExpectationsWs)
    ws.jacobian_static .= view(jacobian, :, ws.backward_nbr .+ ws.current_indices[ws.static_indices])
    geqrf_core!(ws.jacobian_static, ws.qr_ws)
    ormqr_core!('L', ws.jacobian_static', jacobian, ws.qr_ws)
end

"""
Computes the solution for the static variables:
G_y,static = -B_s,s^{-1}(A_s*Gy,fwrd*Gs + B_s,d*Gy,dynamic + C_s) 
""" 
function add_static!(results::LinearRationalExpectationsResults,
                     jacobian::Matrix{Float64},
                     ws::LinearRationalExpectationsWs)
    # static rows are at the top of the QR transformed Jacobian matrix
    i_static = 1:ws.static_nbr
    # B_s,s
    ws.b10 .= view(jacobian, i_static, ws.backward_nbr .+ ws.static_indices)
    # B_s,d
    ws.b11 .= view(jacobian, i_static, ws.backward_nbr .+ ws.dynamic_indices)
    # A_s
    ws.temp1 .= view(jacobian, i_static, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
    # C_s
    ws.temp2 .= view(jacobian, i_static, 1:ws.backward_nbr)
    # Gy,fwrd
    ws.temp3 .= view(results.g1_1, ws.forward_indices, :)
    # Gy,dynamic
    ws.temp4 .= view(results.g1_1, ws.dynamic_indices, :)
    # ws.temp2 = B_s,d*Gy.dynamic + C_s
    mul!(ws.temp2, ws.b11, ws.temp4, 1.0, 1.0)
    mul!(ws.temp6, ws.temp1, ws.temp3)
    mul!(ws.temp2, ws.temp6, results.gs1, -1.0, -1.0)
    # ws.temp3 = S\ws.temp3
    linsolve_core!(ws.b10, ws.temp2, ws.linsolve_static_ws)
    println("ws.temp2")
    display(ws.temp2)
    for i = 1:ws.backward_nbr
        for j=1:ws.static_nbr
            results.g1[ws.static_indices[j],i] = ws.temp2[j,i]
        end
    end
end

function get_abc!(ws::LinearRationalExpectationsWs, jacobian::AbstractMatrix{Float64})
    i_rows = (ws.static_nbr+1):ws.endogenous_nbr
    fill!(ws.a, 0.0)
    fill!(ws.b, 0.0)
    fill!(ws.c, 0.0)
    ws.a[:, ws.forward_indices_d] .= view(jacobian, i_rows, ws.backward_nbr .+ ws.current_nbr .+ (1:ws.forward_nbr))
    ws.b[:, ws.current_dynamic_indices_d] .= view(jacobian, i_rows, ws.backward_nbr .+ ws.current_dynamic_indices)
    ws.c[:, ws.backward_indices_d] .= view(jacobian, i_rows, 1:ws.backward_nbr)
end

function get_de!(ws::LinearRationalExpectationsWs, jacobian::AbstractMatrix{Float64})
    n1 = ws.backward_nbr + ws.forward_nbr - ws.both_nbr
    fill!(ws.d, 0.0)
    fill!(ws.e, 0.0)
    i_rows = (ws.static_nbr + 1):ws.endogenous_nbr
    ws.d[1:n1, ws.icolsD] .= jacobian[i_rows, ws.jcolsD]
    ws.e[1:n1, ws.icolsE] .= -jacobian[i_rows, ws.jcolsE]
    u = Matrix{Float64}(I, ws.both_nbr, ws.both_nbr)                                    
    i_rows = n1 .+ (1:ws.both_nbr)
    ws.d[i_rows, ws.colsUD] .= u
    ws.e[i_rows, ws.colsUE] .= u
end

"""
Computes LU decomposition of A*G + B
"""
function make_lu_AGplusB!(AGplusB, A, G, B, ws)
    fill!(AGplusB, 0.0)
    vAGplusB = view(AGplusB, :, ws.current_indices)
    vAGplusB .= B
    ws.temp3 .= view(G, ws.forward_indices, :)
    mul!(ws.temp7, A, ws.temp3,)
    vAGplusB = view(AGplusB, :, ws.backward_indices)
    vAGplusB .+= ws.temp7
    LinSolveAlgo.lu!(AGplusB, ws.AGplusB_linsolve_ws)
end

function solve_for_derivatives_with_respect_to_shocks!(results::LinearRationalExpectationsResults, jacobian::AbstractMatrix, ws::LinearRationalExpectationsWs)
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
        linsolve_core_no_lu!(ws.AGplusB, results.g1_2, ws.AGplusB_linsolve_ws)
#        end
    end
end

function first_order_solver!(results::LinearRationalExpectationsResults,
                             algo::String,
                             jacobian::Matrix,
                             options,
                             ws::LinearRationalExpectationsWs)
    if algo == "CR"
        cyclic_reduction!(ws.x, ws.c, ws.b, ws.a, ws.solver_ws, options["cyclic_reduction"]["tol"], 100)
        for i = 1:ws.backward_nbr
            for j = 1:ws.backward_nbr
                results.gs1[j, i] = ws.x[ws.backward_indices_d[j], ws.backward_indices_d[i]]
            end
            for j = 1:(ws.endogenous_nbr - ws.static_nbr)
                results.g1[ws.dynamic_indices[j],i] = ws.x[j, ws.backward_indices_d[i]]
            end
        end
    elseif algo == "GS"
        gs_solver!(ws.solver_ws, ws.d, ws.e, ws.backward_nbr, options["generalized_schur"]["criterium"])
        results.gs1 .= ws.solver_ws.g1
        for i = 1:ws.backward_nbr
            for j = 1:ws.backward_nbr
                x = ws.solver_ws.g1[j,i]
                results.g1[ws.backward_indices[j],i] = x
            end
            for j = 1:(ws.forward_nbr - ws.both_nbr)
                results.g1[ws.purely_forward_indices[j], i] =
                    ws.solver_ws.g2[ws.icolsE[ws.backward_nbr + j] - ws.backward_nbr, i]
            end
        end
    else
        error("Algorithm $algo not recognized")
    end
    if ws.static_nbr > 0
        add_static!(results, jacobian, ws)
    end
    A = view(jacobian, :, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
    B = view(jacobian, :, ws.backward_nbr .+ ws.current_indices)
    make_lu_AGplusB!(ws.AGplusB, A, results.g1_1, B, ws)        
    solve_for_derivatives_with_respect_to_shocks!(results, jacobian, ws)
end


