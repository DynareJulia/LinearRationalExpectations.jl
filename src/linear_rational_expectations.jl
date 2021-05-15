using LinearAlgebra
using FastLapackInterface
using FastLapackInterface.LinSolveAlgo
using FastLapackInterface.QrAlgo
using PolynomialMatrixEquations
#using SolveEyePlusMinusAkronB: EyePlusAtKronBWs, generalized_sylvester_solver!

using LinearAlgebra.BLAS

struct LinearRationalExpectationsWs
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
    linsolve_static_ws::LinSolveWs
    AGplusB_linsolve_ws::LinSolveWs
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
        forward_indices_d = findall(in(forward_indices), dynamic_indices)
        backward_indices_d = findall(in(backward_indices), dynamic_indices)
        current_dynamic_indices_d = findall(in(current_dynamic_indices), dynamic_indices)
        current_dynamic_indices_d_j = backward_nbr .+ findall(in(dynamic_indices), current_indices)
        exogenous_indices = backward_nbr + current_nbr + forward_nbr .+ (1:exogenous_nbr)
        if static_nbr > 0
            jacobian_static = Matrix{Float64}(undef, endogenous_nbr, static_nbr)
            qr_ws = QrWs(jacobian_static)
            static_indices_j = backward_nbr .+ [findfirst(isequal(x), current_indices) for x in static_indices] 
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
            current_dynamic_indices_d, current_dynamic_indices_d_j,
            exogenous_indices, static_indices_j,
            static_nbr, dynamic_nbr, forward_nbr, backward_nbr,
            both_nbr, current_nbr, jacobian_static, qr_ws, solver_ws,
            a, b, c, d, e, x,
            ghx, gx, hx, temp1, temp2, temp3, temp4, temp5, temp6, temp7,
            temp8, temp9, b10, b11, icolsD, icolsE, jcolsD, jcolsE,
            colsUD, colsUE, AGplusB, linsolve_static_ws, AGplusB_linsolve_ws)
    end
end

struct LinearRationalExpectationsResults
    g1::Matrix{Float64}  # full approximation
    gs1::Matrix{Float64} # state transition matrices
    g1_1::SubArray # solution first order derivatives w.r. to state variables
     g1_2::SubArray # solution first order derivatives w.r. to current exogenous variables
#    g1_3::SubArray # solution first order derivatives w.r. to lagged exogenous variables
    
    function LinearRationalExpectationsResults(endogenous_nbr::Int64,
                                               exogenous_nbr::Int64,
                                               backward_nbr::Int64)
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
function remove_static!(jacobian::Matrix{Float64},
                        ws::LinearRationalExpectationsWs)
    ws.jacobian_static .= view(jacobian, :, ws.static_indices_j)
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
    # B_s,s
    fill!(ws.b10, 0.0)
    vj = view(jacobian, 1:ws.static_nbr, ws.static_indices_j)
    copy!(ws.b10, vj)
    # B_s,d
    vb = view(ws.b11, :, ws.current_dynamic_indices_d)
    vj = view(jacobian, 1:ws.static_nbr, ws.current_dynamic_indices_d_j)
    copy!(vb, vj)
    # A_s
    vj = view(jacobian, 1:ws.static_nbr, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
    copy!(ws.temp1, vj) 
    # C_s
    vj = view(jacobian, 1:ws.static_nbr, 1:ws.backward_nbr)
    copy!(ws.temp2, vj)
    # Gy.fwrd
    vg = view(results.g1_1, ws.forward_indices, 1:ws.backward_nbr)
    copy!(ws.temp3, vg)
    # Gy.dynamic
    vg = view(results.g1_1, ws.dynamic_indices, 1:ws.backward_nbr)
    copy!(ws.temp4, vg)
    # ws.temp2 = B_s,d*Gy.dynamic + C_s
    mul!(ws.temp2, ws.b11, ws.temp4, 1.0, 1.0)
    # ws.temp6 = A_s*Gy.fwrd*gs1
    mul!(ws.temp6, ws.temp1, ws.temp3)
    mul!(ws.temp2, ws.temp6, results.gs1, -1.0, -1.0)
    # ws.temp3 = B_s,s\ws.temp2
    linsolve_core!(ws.b10, ws.temp2, ws.linsolve_static_ws)
    vg = view(results.g1, ws.static_indices, 1:ws.backward_nbr)
    copy!(vg, ws.temp2)
end

function get_abc!(ws::LinearRationalExpectationsWs,
                  jacobian::AbstractMatrix{Float64})
    fill!(ws.a, 0.0)
    fill!(ws.b, 0.0)
    fill!(ws.c, 0.0)

    va = view(ws.a, :, ws.forward_indices_d)
    vj = view(jacobian, ws.static_nbr .+ (1:ws.dynamic_nbr),
              ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
    copy!(va, vj)
    vb = view(ws.b, :, ws.current_dynamic_indices_d)
    vj = view(jacobian, ws.static_nbr .+ (1:ws.dynamic_nbr),
              ws.current_dynamic_indices_d_j)
    copy!(vb, vj)
    vc = view(ws.c, :, ws.backward_indices_d)
    vj = view(jacobian, ws.static_nbr .+ (1:ws.dynamic_nbr),
              1:ws.backward_nbr)
    copy!(vc, vj)
end

function get_de!(ws::LinearRationalExpectationsWs,
                 jacobian::AbstractMatrix{Float64})
    n1 = ws.backward_nbr + ws.forward_nbr - ws.both_nbr
    fill!(ws.d, 0.0)
    fill!(ws.e, 0.0)
    vd = view(ws.d, 1:ws.dynamic_nbr, ws.icolsD)
    vj = view(jacobian, ws.static_nbr .+ (1:ws.dynamic_nbr),
              ws.jcolsD)
    copy!(vd, vj)
    ve = view(ws.e, 1:ws.dynamic_nbr, ws.icolsE)
    vj = view(jacobian, ws.static_nbr .+ (1:ws.dynamic_nbr),
              ws.jcolsE)
    ve .= .-vj
    for i = 1:ws.both_nbr
        k = ws.dynamic_nbr + i
        m = ws.colsUD[i]
        ws.d[k, m] = 1.0
    end
    for i = 1:ws.both_nbr
        k = ws.dynamic_nbr + i
        m = ws.colsUE[i]
        ws.e[k, m] = 1.0
    end
end

"""
Computes LU decomposition of A*G + B
"""
function make_lu_AGplusB!(AGplusB::AbstractMatrix{Float64},
                          A::AbstractMatrix{Float64},
                          G::AbstractMatrix{Float64},
                          B::AbstractMatrix{Float64},
                          ws::LinearRationalExpectationsWs)
    fill!(AGplusB, 0.0)
    vAG = view(AGplusB, :, ws.current_indices)
    copy!(vAG, B)
    vG = view(G, ws.forward_indices, :)
    copy!(ws.temp3, vG)
    mul!(ws.temp7, A, ws.temp3)
    vAG = view(AGplusB, :, ws.backward_indices)
    vAG .+= ws.temp7
    LinSolveAlgo.lu!(AGplusB, ws.AGplusB_linsolve_ws)
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
        linsolve_core_no_lu!(ws.AGplusB, results.g1_2, ws.AGplusB_linsolve_ws)
#        end
    end
end

function first_order_solver!(results::LinearRationalExpectationsResults,
                             algo::String,
                             jacobian::AbstractMatrix{Float64},
                             options::Dict{String, Any},
                             ws::LinearRationalExpectationsWs)
    remove_static!(jacobian, ws)
    if algo == "CR"
        get_abc!(ws, jacobian)
        cyclic_reduction!(ws.x, ws.c, ws.b, ws.a, ws.solver_ws, options["cyclic_reduction"]["tol"], 100)
        vg = view(results.gs1, :, 1:ws.backward_nbr)
        vx = view(ws.x, ws.backward_indices_d, ws.backward_indices_d)
        copy!(vg, vx)
        vg = view(results.g1,ws.dynamic_indices, 1:ws.backward_nbr)
        vx = view(ws.x, :, ws.backward_indices_d)
        copy!(vg, vx)
    elseif algo == "GS"
        get_de!(ws, jacobian)
        gs_solver!(ws.solver_ws, ws.d, ws.e, ws.backward_nbr, options["generalized_schur"]["criterium"])
        results.gs1 .= ws.solver_ws.g1
        vs = view(ws.solver_ws.g1, 1:ws.backward_nbr, 1:ws.backward_nbr)
        vr = view(results.g1, ws.backward_indices,1:ws.backward_nbr)
        copy!(vr,vs)
        vs = view(ws.solver_ws.g2,
                  ws.icolsE[ws.backward_nbr .+ (1:(ws.forward_nbr - ws.both_nbr))]
                  .-ws.backward_nbr, :)
        vr = view(results.g1, ws.purely_forward_indices, 1:ws.backward_nbr)
        copy!(vr,vs)
    else
        error("Algorithm $algo not recognized")
    end
    if ws.static_nbr > 0
        add_static!(results, jacobian, ws)
    end
    #    A = view(jacobian, :, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
    #    B = view(jacobian, :, ws.backward_nbr .+ ws.current_indices)
    vt = view(ws.temp8, :, 1:ws.forward_nbr)
    vj = view(jacobian, :, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr))
    copy!(vt, vj)
    vt = view(ws.temp9, :, 1:ws.current_nbr)
    vj = view(jacobian, :, ws.backward_nbr .+ (1:ws.current_nbr))
    copy!(vt, vj)
    make_lu_AGplusB!(ws.AGplusB, ws.temp8, results.g1_1, ws.temp9, ws)        
    solve_for_derivatives_with_respect_to_shocks!(results, jacobian, ws)
end


