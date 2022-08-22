using LinearAlgebra
using FastLapackInterface

struct LyapdWs
    AA::Matrix{Float64}
    AAtemp::Matrix{Float64}
    AA2::Matrix{Float64}
    AA2temp::Matrix{Float64}
    BB::Matrix{Float64}
    temp1::Matrix{Float64}
    XX::Vector{Float64}
    nonstationary_variables::Vector{Bool}
    nonstationary_trends::Vector{Bool}
    dgees_ws::SchurWs
    linsolve_ws1::LUWs
    linsolve_ws2::LUWs
    function LyapdWs(n::Int64)
        AA = Matrix{Float64}(undef, n, n)
        AAtemp = similar(AA, n, n)
        AA2 = similar(AA, 2*n, 2*n)
        AA2temp = similar(AA2)
        BB = similar(AA)
        temp1 = similar(AA, n, n)
        XX = similar(AA, 2*n)
        nonstationary_variables = Vector{Bool}(undef, n)
        nonstationary_trends = Vector{Bool}(undef, n)
        dgees_ws = SchurWs(AA)
        linsolve_ws1 = LUWs(n)
        linsolve_ws2 = LUWs(2*n)
        new(AA, AAtemp, AA2, AA2temp, BB, temp1, XX, nonstationary_variables,
            nonstationary_trends, dgees_ws, linsolve_ws1,
            linsolve_ws2)
    end
end

function solve_one_row!(X::AbstractMatrix{Float64},
                        A::AbstractMatrix{Float64},
                        B::AbstractMatrix{Float64},
                        n::Int64, row::Int64, ws::LyapdWs)
    α = A[row, row]
    vA = view(A, 1:row, 1:row)
    vAA = view(ws.AAtemp, 1:row, 1:row)
    copy!(vAA, vA)
    lmul!(α, vAA)
    vAA .-= I(row)
    #exploiting symmetry to have row vectors
    vX = view(X, 1:row, row)
    vB = view(B, 1:row, row)
    vX .= .-vB
    lu_t = LU(factorize!(ws.linsolve_ws1, vAA)...)
    ldiv!(lu_t, vX)
    j = n*(row - 1) + 1
    for i in row:n:row*n-1
        X[i] = X[j]
        j += 1
    end
end

function solve_two_rows!(X::AbstractMatrix{Float64},
                         A::AbstractMatrix{Float64},
                         B::AbstractMatrix{Float64},
                         n::Int64, row::Int64, ws::LyapdWs)
    @inbounds begin
        α11, α21, α12, α22 = A[(row - 1):row, (row - 1):row]
        l2 = 1
        for l1 in 1:row
            k2 = 1
            for k1 in 1:row
                ws.AA2[k2, l2] = α11*A[k1, l1]
                ws.AA2[k2  + 1, l2] = α21*A[k1, l1]
                ws.AA2[k2, l2 + 1] = α12*A[k1, l1]
                ws.AA2[k2 + 1, l2 + 1] = α22*A[k1, l1]
                k2 += 2
            end
            ws.XX[l2] = -B[row - 1, l1]
            ws.XX[l2 + 1] = -B[row, l1]
            l2 += 2
        end

        # TODO: cleanup!
        view(ws.AA2, 1:2*row, 1:2*row) .-= I(2*row)
        vAA2 =  view(ws.AA2temp, 1:2*row, 1:2*row)
        vAA2 .= view(ws.AA2, 1:2*row, 1:2*row)
        lu_t = LU(factorize!(ws.linsolve_ws2, vAA2)...)
        vXX = view(ws.XX, 1:2*row)
        ldiv!(lu_t, vXX)
        j = n*(row - 2) + 1
        m = 1
        for i in row-1:n:row*n
            X[i] = vXX[m]
            X[i + 1] = vXX[m+1]
            X[j] = X[i]
            X[j + n] = X[i+1]
            j += 1
            m += 2
        end
    end
end

"""
    function lyapd(Σ, A, B, ws) solves equation Σ - A*Σ*A' = B
"""
function extended_lyapd!(Σ::AbstractMatrix{Float64},
                         A::AbstractMatrix{Float64},
                         B::AbstractMatrix{Float64},
                         ws::LyapdWs)
    n = size(A, 1)
    copy!(ws.AA, A)
    factorize!(ws.dgees_ws, 'V', ws.AA, select = (wr, wi) -> wr^2 + wi^2 > 1-1e-6)
    mul!(ws.temp1, transpose(ws.dgees_ws.vs), B)
    mul!(ws.BB, ws.temp1, ws.dgees_ws.vs)

    extended_lyapd_core!(Σ, ws.AA, ws.BB, ws)
    mul!(ws.temp1, ws.dgees_ws.vs, Σ)
    mul!(Σ, ws.temp1, ws.dgees_ws.vs')
    
    for i = 1:n
        if ws.nonstationary_trends[i]
            for j = 1:n
                if abs(ws.dgees_ws.vs[j, i]) > 1e-10
                    ws.nonstationary_variables[j] = true
                end
            end
        end
    end
    
    for i = 1:n
        if  ws.nonstationary_variables[i]
            for j = i:n
                Σ[j, i] = NaN
                Σ[i, j] = NaN                    
            end
        else
            for j = i:n
                if  ws.nonstationary_variables[j]
                    Σ[j, i] = NaN
                    Σ[i, j] = NaN
                end
            end
        end
    end
end

function extended_lyapd_core!(Σ::AbstractMatrix{Float64},
                              A::AbstractMatrix{Float64},
                              B::AbstractMatrix{Float64},
                              ws::LyapdWs)
    fill!(Σ, 0.0)
    fill!(ws.nonstationary_trends, false)
    fill!(ws.nonstationary_variables, false)
    n = size(A, 1)
    row = n
    while row >= 1
        if row == 1 || A[row, row - 1] == 0
            if A[row, row] > 1 - 1e-6
                ws.nonstationary_trends[row] = true
            else 
                solve_one_row!(Σ, A, B, n, row, ws)
                vB = view(B, 1:row - 1, 1:row - 1)
                vtemp = view(ws.temp1, 1:row - 1, 1)
                vΣ = view(Σ, 1:row -1 , row)
                vA = view(A, 1:row - 1, 1:row - 1)
                mul!(vtemp, vA, vΣ)
                vA = view(A, 1:row - 1, row)
                mul!(vB, vtemp, vA', 1.0, 1.0)
                vΣ = view(Σ, 1:row, row)
                vA = view(A, 1:row - 1, 1:row)
                mul!(vtemp, vA, vΣ)
                vA = view(A, 1:row - 1, row)
                vB .= vB .+ vA.*vtemp'
            end
            row -= 1
        else
            a = A[row, row]
            if a*a + A[row, row - 1]*A[row - 1, row] > 1 - 2e-6
                ws.nonstationary_trends[row] = true
                ws.nonstationary_trends[row - 1] = true
            else 
                solve_two_rows!(Σ, A, B, n, row, ws)
                vB = view(B, 1:row - 2, 1:row - 2)
                vtemp = view(ws.temp1, 1:row - 2, 1:2)
                vΣ = view(Σ, 1:row - 2, row - 1:row)
                vA = view(A, 1:row - 2, 1:row - 2)
                mul!(vtemp, vA, vΣ)
                vA = view(A, 1:row - 2, row - 1:row)
                mul!(vB, vtemp, vA', 1.0, 1.0)
                vΣ = view(Σ, 1:row, row - 1:row)
                vA = view(A, 1:row - 2, 1:row)
                mul!(vtemp, vA, vΣ)
                vA = view(A, 1:row - 2, row - 1:row)
                mul!(vB, vA, vtemp', 1.0, 1.0)
            end
            row -= 2
        end
    end
end
                                 
function is_stationary(ws::LyapdWs)
    return !any(ws.nonstationary_trends)
end
