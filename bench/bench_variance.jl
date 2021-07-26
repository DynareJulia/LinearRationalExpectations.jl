using BandedMatrices
using BenchmarkTools
using LinearAlgebra
using LinearRationalExpectations
using Test
function make_orthogonal_matrix(n)
    W = randn(n, n)
    w = view(W, :,1)
    A = w/norm(w)
    for i=2:n
        w = view(W, :, i)
        v = (I - A*transpose(A))*w
        v = v/norm(v)
        A = hcat(A, v)
    end
    return A
end

n = 10
A = make_orthogonal_matrix(n) 
@test A'*A ≈ I(n)

function make_complex2x2!(V::AbstractMatrix{Float64}, W::AbstractVector{Float64})
    a = W[1]
    b = W[2]
    c = W[3]
    V[1, 1] = V[2, 2] = a
    V[2, 1] = b
    x = abs(c)
    V[1, 2] = (b < 0) ? x : -x
end

function make_stationary_matrix_complex_roots(n::Int64; p::Float64 = 1/5, m::Float64 = 0.99)
    Q = make_orthogonal_matrix(n)
    values = rand(-m:0.001:m, n)
    V = BandedMatrix(Pair(-1, zeros(n - 1)), Pair(0, values), Pair(1, zeros(n - 1)))
    pn = Integer(floor(p*n))
    KK = rand(1:n, pn)
    W = randn(pn, 3)
    mc = maximum(abs.(view(W, :, 3)))
    @views W[:, 3] ./= W[:, 3]/mc
    @views for (i, k) in enumerate(KK)
        if (k == 1 ||
            (k < n && V[k, k-1] == 0 && V[k, k+1] == 0))
            make_complex2x2!(V[k:k+1, k:k+1], W[i, :])
        end
    end
    return Q*V
end

function make_ABΣe!(n, ns, nns, nx)
    state_nbr = ns + nns
    nonstate_nbr = n - nns - ns
    A1a = triu(randn(nns, nns), 1) + I
    A1b = randn(nns, ns)
    A2a = randn(nonstate_nbr, ns)
    A2b = make_stationary_matrix_complex_roots(ns)
    A = hcat(zeros(n, nonstate_nbr),
             vcat(hcat(zeros(nonstate_nbr, nns),
                       A2a),
                  hcat(A1a, A1b),
                  hcat(zeros(ns, nns),
                       A2b)))
    B = rand(n, nx)
    Σe = randn(nx, nx)
    Σe = Σe*transpose(Σe)
    return (A, B, Σe)
end

n = 100
ns = 50
nns = 20
nx = 30
state_nbr = ns + nns
nonstate_nbr = n - nns - ns
backward_indices = nonstate_nbr + 1:n
non_backward_indices = 1:nonstate_nbr

(A, B, Σe) = make_ABΣe!(n, ns, nns, nx)

A1 = A[backward_indices, backward_indices]
A2 = A[non_backward_indices, backward_indices]


B1 = B[backward_indices, :]
B2 = B[non_backward_indices, :]

Σyss = Matrix{Float64}(undef, state_nbr, state_nbr)

@btime wsl = LinearRationalExpectations.LyapdWs(state_nbr)
wsl = LinearRationalExpectations.LyapdWs(state_nbr)
@btime LinearRationalExpectations.extended_lyapd!(Σyss, A1, B1*Σe*transpose(B1), wsl)

Σy = zeros(n, n)

algo = "GS"
forward_indices = collect(1:nonstate_nbr - 1)
current_indices = collect(1:n)
both_indices = Vector{Int64}(undef, 0)
static_indices = [nonstate_nbr]
lre_ws = LinearRationalExpectationsWs(algo,
                                      n,
                                      nx,
                                      0,
                                      forward_indices,
                                      current_indices,
                                      collect(backward_indices),
                                      both_indices,
                                      static_indices)

@btime ws = LinearRationalExpectations.VarianceWs(n,
                                                  state_nbr,
                                                  nx,
                                                  lre_ws)
ws = LinearRationalExpectations.VarianceWs(n,
                                           state_nbr,
                                           nx,
                                           lre_ws)
@btime LinearRationalExpectations.compute_variance!(Σy, A1, A2, B1, B2, Σe, ws)
