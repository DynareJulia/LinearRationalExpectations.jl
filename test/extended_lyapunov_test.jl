using LinearAlgebra
using Random
using Test
Random.seed!(123)
n = 5
Xorig = randn(n, n)
A = UpperTriangular(rand(n, n))
B = randn(n, n)
B = B'*B
ws = LyapdWs(n)
for row = 1:n
    X = copy(Xorig)
    LinearRationalExpectations.solve_one_row!(X, A, B, n, row, ws)
    @test X[row, 1:row] ≈ -(A[row, row]*A[1:row, 1:row] - I(row))\B[row, 1:row]
    @test X[1:row, row] == X[row, 1:row]
end

n = 5
Xorig = randn(n, n)
A = zeros(n, n) .+ UpperTriangular(rand(n, n))
B = randn(n, n)
B = B'*B
ws = LyapdWs(n)
for row = 2:n
    A[row, row - 1] = 0.5
    X = copy(Xorig)
    Borig = copy(B)
    LinearRationalExpectations.solve_two_rows!(X, A, B, n, row, ws)
    @test B == Borig
    @test ws.AA2[1:2*row, 1:2*row] == kron(A[1:row, 1:row], A[(row - 1):row, (row - 1):row]) - I
    @test vec(X[(row-1):row, 1:row]) ≈ -(kron(A[1:row, 1:row], A[(row - 1):row, (row - 1):row]) - I(2*row))\vec(B[(row-1):row, 1:row])
    @test X[1:row, row-1] == X[row-1, 1:row]
    @test X[1:row, row] == X[row, 1:row]
    A[row, row - 1] = 0
end

for i = 1:100
    s = Int(floor(1000*rand()))
    Random.seed!(s)
    local n = 5
    X1 = randn(n, n)
    X = X1'*X1
    local A = randn(n, n)
    F = eigen(A)
    if maximum(abs.(F.values)) > 1.0
        continue
    end
    local B = X - A*X*A'
    local ws = LyapdWs(n)
    Σ = similar(X)
    extended_lyapd!(Σ, A, B, ws)
    @test Σ ≈ X
end

# with unit roots
for i = 1:100
    s = Int(floor(1000*rand()))
    Random.seed!(s)
    local n = 2
    X1 = randn(n, n)
    X = X1'*X1
    local A = randn(n, n)
    F = eigen(A)
    eigenvalues = diagm(rand(n))
    local A = real(F.vectors*eigenvalues*inv(F.vectors))
    m = 2
    AA = vcat(hcat([ 1.0 -0.5 0; 0 1.0 0], randn(m, n)),
              [1 -1 0 0 0],
              hcat(zeros(n, m+1), A))
    if abs(det(AA))  < 1e-12
        continue
    end
    local B = X - A*X*A'
    BB = vcat(hcat(I(m), zeros(m, n+1)),
              [0 0 0.2 0 0],
              hcat(zeros(n, m+1), B))

    local ws = LyapdWs(n + m + 1)
    Σ = Matrix{Float64}(undef, n+m+1, n+m+1)
    extended_lyapd!(Σ, AA, BB, ws)
    @test Σ[m + 1 .+ (1:n), m + 1 .+ (1:n)] ≈ X
end
