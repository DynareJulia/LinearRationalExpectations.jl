using LinearRationalExpectations
using Test

#=
Test 1:
Consider the following lead lag incidence matrix:
 0   1  2  3  0   4   0
 5   6  7  8  9  10  11
 0  12  0  0  0   0  13
=#

ws = LinearRationalExpectationsWs("GS",
                                  7, # endogenous_nbr,
                                  2, #exogenous_nbr,
                                  0, #exogenous_deterministic_nbr,
                                  [2, 7], #forward indices
                                  collect(1:7), #current indices
                                  [2, 3, 4, 6], #backward indices
                                  [2], # both indices
                                  [5], # static indices
                                  )
@test ws.icolsD == 1:6
@test ws.jcolsD == [6, 7, 8, 10, 12, 13]
@test ws.icolsE == [1, 2, 3, 4, 6]
@test ws.jcolsE == [1, 2, 3, 4, 11]


#=
TO BE FIXED without reference to Dynare.jl

using FastLapackInterface
using FastLapackInterface.LinSolveAlgo
using LinearAlgebra
using LinearRationalExpectations
using MAT
using Test

struct Cycle_Reduction
    tol
end

cr_opt = Cycle_Reduction(1e-8)

struct Generalized_Schur
    criterium
end

gs_opt = Generalized_Schur(1+1e-6)

struct Options
    cycle_reduction
    generalized_schur
end

options = Options(cr_opt,gs_opt)
        
file = matopen("models/example1/example1_results.mat")
oo_ = read(file,"oo_")

#=
function test_model(endo_nbr,lead_lag_incidence, param_nbr)
    m = Model("models/example1/example1", endo_nbr,lead_lag_incidence, 2, 0, 0, param_nbr)

    @test m.DErows1 == 1:5
    @test m.DErows2 == 6:6
    @test m.n_dyn == 6
    @test m.icolsD == [1, 2, 4, 5, 6]
    @test m.jcolsD == [6, 7, 10, 11, 12]
    @test m.icolsE == [1, 2, 3, 4, 5, 6]
    @test m.jcolsE == [1, 2, 3, 4, 5, 9]
    @test m.colsUD == 3:3
    @test m.colsUE == 6:6
end
=#

function test_getDE(endo_nbr, lead_lag_incidence, jacobian, param_nbr)
    m = Model("models/example1/example1", endo_nbr,lead_lag_incidence, 2, 0, 0, param_nbr)
    ws = LinearRationalExpectationsWs("GS",
                                      m.endogenous_nbr,
                                      m.exogenous_nbr,
                                      m.exogenous_deterministic_nbr,
                                      m.i_fwrd_b,
                                      m.i_current,
                                      m.i_bkwrd_b,
                                      m.i_both,
                                      m.i_static)
    LinearRationalExpectations.remove_static!(jacobian, ws)
    @test norm(jacobian[m.n_static+1:end, m.p_static]
               - zeros(m.endogenous_nbr - m.n_static, m.n_static), Inf) < 1e-15
    LinearRationalExpectations.get_de!(ws, jacobian)
    Dtarget = zeros(6,6)
    Etarget = zeros(6,6)
    Dtarget[1:5, :] = jacobian[2:6,[6, 7, 9, 10, 11, 12]]
    Dtarget[6, 3] = 1
    Etarget[1:5, 1:5] = -jacobian[2:6,[1, 2, 3, 4, 5]]
    Etarget[6, 6] = 1 
    @test ws.d == Dtarget
    @test ws.e == Etarget 
end

function test_make_lu_AGplusB(endo_nbr, lead_lag_incidence, algo, jacobian, param_nbr)
    m = Model("models/example1/example1", endo_nbr,lead_lag_incidence, 2, 0, 0, param_nbr)
    ws = LinearRationalExpectationsWs(algo,
                                      m.endogenous_nbr,
                                      m.exogenous_nbr,
                                      m.exogenous_deterministic_nbr,
                                      m.i_fwrd_b,
                                      m.i_current,
                                      m.i_bkwrd_b,
                                      m.i_both,
                                      m.i_static)
    results = ResultsPerturbationWs(1, m.endogenous_nbr, m.exogenous_nbr, m.n_states)
    A = jacobian[:, 10:12]
    @test view(jacobian, :, ws.backward_nbr + ws.current_nbr .+ (1:ws.forward_nbr)) == A
    B = jacobian[:, 4:9]
    @test view(jacobian, :, ws.backward_nbr .+ ws.current_indices) == B
    results.g1_1 .= randn(6, 3)
    target = zeros(ws.endogenous_nbr, ws.endogenous_nbr)
    target .= B
    target[:, [3, 4, 6]] .+= A*results.g1_1[[1, 2, 6], :]
    LinearRationalExpectations.make_lu_AGplusB!(ws.AGplusB, A, results.g1_1, B, ws)
    @test ws.AGplusB == target
    L, U = lu(ws.AGplusB)
    target = tril(L, -1) .+ U
    @test ws.AGplusB_linsolve_ws.lu ≈ target
end

function test_solver(endo_nbr,
                     lead_lag_incidence,
                     options,
                     algo,
                     jacobian,
                     param_nbr)
    m = Model("models/example1/example1", endo_nbr,lead_lag_incidence, 2, 0, 0, param_nbr)
    ws = LinearRationalExpectationsWs(algo,
                                      m.endogenous_nbr,
                                      m.exogenous_nbr,
                                      m.exogenous_deterministic_nbr,
                                      m.i_fwrd_b,
                                      m.i_current,
                                      m.i_bkwrd_b,
                                      m.i_both,
                                      m.i_static)
    results = ResultsPerturbationWs(1, m.endogenous_nbr, m.exogenous_nbr, m.n_states)
    LinearRationalExpectations.remove_static!(jacobian, ws)
    @test size(jacobian) == (6,14)
    if algo == "GS"
        LinearRationalExpectations.get_de!(ws, jacobian)
        LinearRationalExpectations.first_order_solver!(results, algo, jacobian, options, ws)
    else
        LinearRationalExpectations.get_abc!(ws, jacobian)
        LinearRationalExpectations.first_order_solver!(results, algo, jacobian, options, ws)
    end
    k = dropdims(round.(Int,oo_["dr"]["inv_order_var"]); dims=2)
    res = norm(results.g[1][:, 1:(m.n_bkwrd + m.n_both)]-oo_["dr"]["ghx"][k,:],Inf)
    @test res < 1e-13
    res = norm(results.g[1][:, m.n_bkwrd + m.n_both .+ (1:m.exogenous_nbr)]-oo_["dr"]["ghu"][k,:],Inf)
    @test res < 1e-13
end

function solve_large_model(endo_nbr,lead_lag_incidence,options,algo,jacobian, param_nbr)
    m = Model("models/example1/example1", endo_nbr,lead_lag_incidence, 2, 0, 0, param_nbr)
    ws = LinearRationalExpectationsWs(algo, jacobian, m)
    results = ResultsPerturbationWs(1, m.endogenous_nbr, m.exogenous_nbr, m.n_states)
    @time    first_order_solver(results, ws,algo, jacobian, m, options)
end    

include("models/make_examples.jl")

context = Dynare.Context()
parser("models/example1/example1", context)
lli = context.models[1].lead_lag_incidence
model = context.models[1]
results = context.results.model_results[1]
work = context.work
test_model(6,lli, model.parameter_nbr)
steady_state!(context)
exogenous = zeros(2, model.exogenous_nbr)
get_jacobian_at_steadystate!(work,
                             results.endogenous_steady_state,
                             exogenous,
                             model,
                             2)
test_make_lu_AGplusB(6, lli, "CR", context.work.jacobian, model.parameter_nbr)
test_make_lu_AGplusB(6, lli, "GS", context.work.jacobian, model.parameter_nbr)
test_solver(6, lli, options, "CR", context.work.jacobian, model.parameter_nbr)
test_getDE(6, lli, context.work.jacobian, model.parameter_nbr)
test_solver(6, lli, options, "GS", context.work.jacobian, model.parameter_nbr)
#=
n = 100
lli2, jacobian2 = make_model(n)
fu = zeros(6*n,2*n)
cols = 1
rows = 5
for i=1:n
    fu[rows, cols] = 1
    fu[rows + 1, cols + 1] = 1
    global rows += 6
    global cols += 2
end
jacobian2 = hcat(jacobian2, fu)
println("large model")
solve_large_model(n*6,lli2,options,"CR",jacobian2)
solve_large_model(n*6,lli2,options,"CR",jacobian2)
println("OK")
=#

=#

