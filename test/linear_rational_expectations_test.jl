using Random
using LinearRationalExpectations: n_backward, n_forward, n_both, n_static, n_dynamic, n_current, n_exogenous

# Regular linear rational expectations model
function make_jacobian(ws)
    jacobian_orig = zeros(ws.ids.n_endogenous,
                          n_backward(ws.ids)
                          + n_current(ws.ids)
                          + n_forward(ws.ids)
                          + n_exogenous(ws.ids))
    for i = 1:1000
        Random.seed!(i)
        jacobian = rand(ws.ids.n_endogenous,
                          n_backward(ws.ids)
                          + n_current(ws.ids)
                          + n_forward(ws.ids)
                          + n_exogenous(ws.ids))
        copy!(jacobian_orig, jacobian)
        LRE.remove_static!(jacobian, ws)
        LRE.copy_jacobian!(ws.solver_ws, jacobian)
        F = schur(ws.solver_ws.e, ws.solver_ws.d)
        if count(abs.(F.α ./ F.β) .> 1.0) == n_forward(ws.ids)
            break
        end
    end
    return jacobian_orig
end

algo = "GS"

endogenous_nbr = 10
exogenous_nbr = 3
forward_indices = [2, 3, 5, 7, 9]
current_indices = collect(2:10)
backward_indices = [1, 4, 6, 7, 9]
both_indices = [7, 9]
static_indices = [8, 10]


ws = LinearRationalExpectationsWs(algo,
                                  exogenous_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  static_indices)
@test ws.ids.current_in_static_jacobian == [12, 14]

results = LinearRationalExpectationsResults(endogenous_nbr,
                                            exogenous_nbr,
                                            n_backward(ws.ids))
jacobian = make_jacobian(ws)
jacobian_orig = copy(jacobian)
LRE.remove_static!(jacobian, ws)
FQ = LinearAlgebra.qr(jacobian_orig[:, [12, 14]])
@test jacobian ≈ transpose(FQ.Q)*jacobian_orig

LRE.copy_jacobian!(ws.solver_ws, jacobian)
@test ws.solver_ws.d[1:8, 1] == zeros(8)
@test ws.solver_ws.d[1:8, 2:10]  == jacobian[3:end, [8, 10, 11, 13, 15, 16, 17, 18, 19]]
@test ws.solver_ws.e[1:8, 1:8]  == -jacobian[3:end, [1, 2, 3, 4, 5, 6, 7, 9]]

targetD = zeros(2, 10)
targetD[1, 4] = 1.0
targetD[2, 5] = 1.0
targetE = zeros(2, 10)
targetE[1, 9] = 1.0
targetE[2, 10] = 1.0
@test ws.solver_ws.d[9:10,:] == targetD
@test ws.solver_ws.e[9:10,:] == targetE

d_orig = copy(ws.solver_ws.d)
e_orig = copy(ws.solver_ws.e)
options = LinearRationalExpectationsOptions()
n_back = n_backward(ws.ids)
back_r = 1:n_back
solver_ws = ws.solver_ws
LRE.solve!(solver_ws, options.generalized_schur.criterium)
@test d_orig * vcat(I(n_back), solver_ws.solver_ws.g2[:, back_r])*solver_ws.solver_ws.g1 ≈ e_orig * vcat(I(n_back), solver_ws.solver_ws.g2[:, back_r])

results.gs1 .= solver_ws.solver_ws.g1
for i = back_r
    for j = back_r
        x = solver_ws.solver_ws.g1[j,i]
        results.g1[ws.ids.backward[j],i] = x
    end
    for j = 1:(n_forward(ws.ids) - n_both(ws.ids))
        results.g1[ws.ids.purely_forward[j], i] =
            solver_ws.solver_ws.g2[ws.ids.E_columns.E[n_back + j] - n_back, i]
    end
end

n_stat = n_static(ws.ids)
LinearRationalExpectations.add_static!(results, jacobian, ws)
@test ws.b11[:, 2:8] ≈ jacobian[1:n_stat, [6, 7, 8, 9, 10, 11, 13]]
@test results.g1[ws.ids.forward, back_r] ≈ solver_ws.solver_ws.g2
@test results.g1[ws.ids.backward, back_r] ≈ solver_ws.solver_ws.g1
b10 = jacobian[1:n_stat, [12, 14]]
@test ws.b10 ≈ b10
target = -b10\(jacobian[1:n_stat, 15:19]
               *results.g1_1[ws.ids.forward,:]
               *results.gs1
               + jacobian[1:n_stat, [6, 7, 8, 9, 10, 11, 13]]
               *results.g1_1[[2, 3, 4, 5, 6, 7, 9], :]
               + jacobian[1:n_stat, back_r])
@test results.g1[ws.ids.static, back_r] ≈ target

A = randn(10, 5)
B = randn(10, 9) 
G = randn(endogenous_nbr, length(backward_indices))
lu_t = LU(LinearRationalExpectations.factorize!(ws.AGplusB_linsolve_ws, copy(LinearRationalExpectations.make_AGplusB!(ws.AGplusB, A, G, B, ws)))...)
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
                                  exogenous_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  static_indices)
solver_ws = ws.solver_ws
jacobian = make_jacobian(ws)
jacobian_orig = copy(jacobian)
first_order_solver!(results, jacobian, options, ws)
@test d_orig * vcat(I(n_backward(ws.ids)), solver_ws.solver_ws.g2[:, 1:n_backward(ws.ids)])*solver_ws.solver_ws.g1 ≈ e_orig * vcat(I(n_backward(ws.ids)), solver_ws.solver_ws.g2[:, 1:n_backward(ws.ids)])
A2 = jacobian[:, 15:19]
A1 = jacobian[:, 6:14]
A0 = jacobian[:, 1:5]
@test A2[3:10,:] ≈ d_orig[1:8, 6:10]
@test A1[3:10, [3, 5, 6, 8]] ≈ d_orig[1:8, 2:5]
@test A1[3:10, [1, 2, 4]] ≈ -e_orig[1:8, 6:8]
@test A0[3:10, :] ≈ -e_orig[1:8, 1:5]
@test d_orig[1:8, 1] ≈ zeros(8, 1)
@test e_orig[1:8, 9:10] ≈ zeros(8, 2)
@test results.g1[ws.ids.forward, back_r] ≈ solver_ws.solver_ws.g2
@test results.g1[ws.ids.backward, back_r] ≈ solver_ws.solver_ws.g1
g1_1 = results.g1_1
g1_2 = results.g1_2
gg = g1_1[ws.ids.forward, :]
@test gg ≈ solver_ws.solver_ws.g2
A2 = jacobian_orig[:, 15:19]
A1 = jacobian_orig[:, 6:14]
A0 = jacobian_orig[:, 1:5]
B = jacobian_orig[:, 20:22]

@test (A2*gg*g1_1[ws.ids.backward,:] +
       A1*g1_1[ws.ids.current,:] ≈ -A0 )
@test (A2*gg*g1_2[ws.ids.backward,:] +
       A1*g1_2[ws.ids.current,:] ≈ -B )

                             
algo = "CR"
ws = LinearRationalExpectationsWs(algo,
                                  exogenous_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  static_indices)
LRE.copy_jacobian!(ws.solver_ws,  jacobian)
@test ws.solver_ws.a[:, [2, 3, 5, 7, 8]] == jacobian[3:end, 15:19]
@test ws.solver_ws.b[:, [2, 3, 4, 5, 7, 8]] == jacobian[3:end, [6, 7, 8, 9, 11, 13]]
@test ws.solver_ws.c[:, [1, 4, 6, 7, 8]] == jacobian[3:end, 1:5]

# purely backward model

endogenous_nbr = 10
exogenous_nbr = 3
forward_indices = Int[]
current_indices = collect(1:10)
backward_indices = [1, 4, 6, 7, 9]
both_indices = Int[]
static_indices = [2, 3, 5, 8, 10]
jacobian = randn(endogenous_nbr, length(backward_indices) + endogenous_nbr + exogenous_nbr)


ws = LinearRationalExpectationsWs("GS",
                                  exogenous_nbr,
                                  forward_indices,
                                  current_indices,
                                  backward_indices,
                                  static_indices)
results = LinearRationalExpectationsResults(endogenous_nbr, exogenous_nbr, length(backward_indices))
LRE.first_order_solver!(results, jacobian, options, ws)
@test results.g1_1 ≈ -jacobian[:, 6:15]\jacobian[:, 1:5]
@test results.g1_2 ≈ -jacobian[:, 6:15]\jacobian[:, 16:18]
@test results.gs1 == results.g1_1[backward_indices, :]
