algo = "CR"

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

order = 1
backward_nbr_b = length(backward_indices) + length(both_indices)
results = LinearRationalExpectationsResults(order,
                                            endogenous_nbr,
                                            exogenous_nbr,
                                            backward_nbr_b)

jacobian = randn(endogenous_nbr,
                 length(forward_indices)
                 + length(current_indices)
                 + length(backward_indices)
                 + 2*length(both_indices)
                 + exogenous_nbr)

jacobian_orig = copy(jacobian)
LinearRationalExpectations.remove_static!(jacobian, ws)
FQ = LinearAlgebra.qr(jacobian_orig[:, [12, 14]])
@test jacobian ≈ transpose(FQ.Q)*jacobian_orig

LinearRationalExpectations.get_abc!(ws, jacobian)
@test ws.a[:, [2, 3, 5, 7, 8]] == jacobian[3:end, 15:19]
@test ws.b[:, [2, 3, 4, 5, 7, 8]] == jacobian[3:end, [6, 7, 8, 9, 11, 13]]
@test ws.c[:, [1, 4, 6, 7, 8]] == jacobian[3:end, 1:5]

A = randn(10, 5)
B = randn(10, 9) 
G = randn(endogenous_nbr, length(backward_indices))
LinearRationalExpectations.make_lu_AGplusB!(ws.AGplusB, A, G, B, ws)
AGplusB = zeros(endogenous_nbr, endogenous_nbr)
AGplusB[:, 2:10] = B 
AGplusB[:, [1, 4, 6, 7, 9]] += A*G[[2, 3, 5, 7, 9], :]
@test AGplusB ≈ ws.AGplusB
LU = reshape(ws.AGplusB_linsolve_ws.lu, endogenous_nbr, endogenous_nbr)
U = triu(LU)
L = tril(LU, -1) + diagm(ones(10))
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
