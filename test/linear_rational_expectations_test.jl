using Random

function make_jacobian(ws)
    jacobian_orig = zeros(ws.endogenous_nbr,
                          ws.backward_nbr
                          + ws.current_nbr
                          + ws.forward_nbr
                          + ws.exogenous_nbr)
    for i = 1:1000
        Random.seed!(i)
        jacobian = randn(ws.endogenous_nbr,
                          ws.backward_nbr
                          + ws.current_nbr
                          + ws.forward_nbr
                         + ws.exogenous_nbr)
        copy!(jacobian_orig, jacobian)
        LinearRationalExpectations.remove_static!(jacobian, ws)
        LinearRationalExpectations.get_de!(ws, jacobian)
        F = schur(ws.e, ws.d)
        if count(abs.(F.α ./ F.β) .> 1.0) == ws.forward_nbr
            break
        end
    end
    return jacobian_orig
end

algo = "GS"

endogenous_nbr = 10
exogenous_nbr = 3
exogenous_deterministic_nbr = 0
forward_indices = [2, 3, 5, 7, 9]
current_indices = collect(2:10)
backward_indices = [1, 4, 6, 7, 9]
both_indices = [7, 9]
static_indices = [8, 10]


ws = LinearRationalExpectationsWs(algo,
                                  endogenous_nbr,
                                  exogenous_nbr,
                                  exogenous_deterministic_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  both_indices,
                                  static_indices)
@test ws.static_indices_j == [12, 14]

results = LinearRationalExpectationsResults(endogenous_nbr,
                                            exogenous_nbr,
                                            ws.backward_nbr)
jacobian = make_jacobian(ws)
jacobian_orig = copy(jacobian)
LinearRationalExpectations.remove_static!(jacobian, ws)
FQ = LinearAlgebra.qr(jacobian_orig[:, [12, 14]])
@test jacobian ≈ transpose(FQ.Q)*jacobian_orig

LinearRationalExpectations.get_de!(ws, jacobian)
@test ws.d[1:8, 1] == zeros(8)
@test ws.d[1:8, 2:10]  == jacobian[3:end, [8, 10, 11, 13, 15, 16, 17, 18, 19]]
@test ws.e[1:8, 1:8]  == -jacobian[3:end, [1, 2, 3, 4, 5, 6, 7, 9]]

targetD = zeros(2, 10)
targetD[1, 4] = 1.0
targetD[2, 5] = 1.0
targetE = zeros(2, 10)
targetE[1, 9] = 1.0
targetE[2, 10] = 1.0
@test ws.d[9:10,:] == targetD
@test ws.e[9:10,:] == targetE

d_orig = copy(ws.d)
e_orig = copy(ws.e)
options = LinearRationalExpectationsOptions()
LinearRationalExpectations.PolynomialMatrixEquations.gs_solver!(ws.solver_ws, ws.d, ws.e, ws.backward_nbr, options.generalized_schur.criterium)
@test d_orig * vcat(I(ws.backward_nbr), ws.solver_ws.g2[:, 1:ws.backward_nbr])*ws.solver_ws.g1 ≈ e_orig * vcat(I(ws.backward_nbr), ws.solver_ws.g2[:, 1:ws.backward_nbr])

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

LinearRationalExpectations.add_static!(results, jacobian, ws)
@test ws.b11[:, 2:8] ≈ jacobian[1:ws.static_nbr, [6, 7, 8, 9, 10, 11, 13]]
@test results.g1[ws.forward_indices, 1:ws.backward_nbr] ≈ ws.solver_ws.g2
@test results.g1[ws.backward_indices, 1:ws.backward_nbr] ≈ ws.solver_ws.g1
b10 = jacobian[1:ws.static_nbr, [12, 14]]
@test ws.b10 ≈ b10
target = -b10\(jacobian[1:ws.static_nbr, 15:19]
               *results.g1_1[ws.forward_indices,:]
               *results.gs1
               + jacobian[1:ws.static_nbr, [6, 7, 8, 9, 10, 11, 13]]
               *results.g1_1[[2, 3, 4, 5, 6, 7, 9], :]
               + jacobian[1:ws.static_nbr, 1:ws.backward_nbr])
@test results.g1[ws.static_indices, 1:ws.backward_nbr] ≈ target

A = randn(10, 5)
B = randn(10, 9) 
G = randn(endogenous_nbr, length(backward_indices))
lu_t = LU(LinearRationalExpectations.make_lu_AGplusB!(ws.AGplusB, A, G, B, ws)...)
AGplusB = zeros(endogenous_nbr, endogenous_nbr)
AGplusB[:, 2:10] = B 
AGplusB[:, [1, 4, 6, 7, 9]] += A*G[[2, 3, 5, 7, 9], :]
@test AGplusB ≈ ws.AGplusB
U = triu(lu_t.factors)
L = tril(lu_t.factors, -1) + diagm(ones(10))

function swaprow!(x, i, j)
    for k in axes(x, 2)
        x[i, k], x[j, k] = x[j, k], x[i, k]
    end
end
target = copy(AGplusB)
for i in axes(target, 2)
    swaprow!(target, i, ws.AGplusB_linsolve_ws.ipiv[i])
end
@test target ≈ L*U 

LinearRationalExpectations.solve_for_derivatives_with_respect_to_shocks!(results, jacobian, ws)
g1_2 = -AGplusB\jacobian[:, 20:22]
@test results.g1_2 ≈ g1_2


algo = "GS"
ws = LinearRationalExpectationsWs(algo,
                                  endogenous_nbr,
                                  exogenous_nbr,
                                  exogenous_deterministic_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  both_indices,
                                  static_indices)
jacobian = make_jacobian(ws)
jacobian_orig = copy(jacobian)
first_order_solver!(results, algo, jacobian, options, ws)
@test d_orig * vcat(I(ws.backward_nbr), ws.solver_ws.g2[:, 1:ws.backward_nbr])*ws.solver_ws.g1 ≈ e_orig * vcat(I(ws.backward_nbr), ws.solver_ws.g2[:, 1:ws.backward_nbr])
A2 = jacobian[:, 15:19]
A1 = jacobian[:, 6:14]
A0 = jacobian[:, 1:5]
@test A2[3:10,:] ≈ d_orig[1:8, 6:10]
@test A1[3:10, [3, 5, 6, 8]] ≈ d_orig[1:8, 2:5]
@test A1[3:10, [1, 2, 4]] ≈ -e_orig[1:8, 6:8]
@test A0[3:10, :] ≈ -e_orig[1:8, 1:5]
@test d_orig[1:8, 1] ≈ zeros(8, 1)
@test e_orig[1:8, 9:10] ≈ zeros(8, 2)
@test results.g1[ws.forward_indices, 1:ws.backward_nbr] ≈ ws.solver_ws.g2
@test results.g1[ws.backward_indices, 1:ws.backward_nbr] ≈ ws.solver_ws.g1
g1_1 = results.g1_1
g1_2 = results.g1_2
gg = g1_1[ws.forward_indices, :]
@test gg ≈ ws.solver_ws.g2
A2 = jacobian_orig[:, 15:19]
A1 = jacobian_orig[:, 6:14]
A0 = jacobian_orig[:, 1:5]
B = jacobian_orig[:, 20:22]

@test (A2*gg*g1_1[ws.backward_indices,:] +
       A1*g1_1[ws.current_indices,:] ≈ -A0 )
@test (A2*gg*g1_2[ws.backward_indices,:] +
       A1*g1_2[ws.current_indices,:] ≈ -B )

                             
algo = "CR"
ws = LinearRationalExpectationsWs(algo,
                                  endogenous_nbr,
                                  exogenous_nbr,
                                  exogenous_deterministic_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  both_indices,
                                  static_indices)
LinearRationalExpectations.get_abc!(ws, jacobian)
@test ws.a[:, [2, 3, 5, 7, 8]] == jacobian[3:end, 15:19]
@test ws.b[:, [2, 3, 4, 5, 7, 8]] == jacobian[3:end, [6, 7, 8, 9, 11, 13]]
@test ws.c[:, [1, 4, 6, 7, 8]] == jacobian[3:end, 1:5]


