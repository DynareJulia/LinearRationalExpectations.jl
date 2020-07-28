using LinearAlgebra
using FastLapackInterface
using FastLapackInterface.LinSolveAlgo
using FastLapackInterface.SchurAlgo

struct LyapdWs
    AA::Matrix{Float64}
    AAtemp::Matrix{Float64}
    AA2::Matrix{Float64}
    BB::Matrix{Float64}
    temp1::Matrix{Float64}
    XX::Vector{Float64}
    nonstationary_variables::Vector{Bool}
    dgees_ws::DgeesWs
    linsolve_ws1::LinSolveWs
    linsolve_ws2::LinSolveWs
    function LyapdWs(n)
        AA = Matrix{Float64}(undef, n, n)
        AAtemp = Matrix{Float64}(undef, n, n)
        AA2 = Matrix{Float64}(undef, 2*n, 2*n)
        BB = similar(AA)
        temp1 = Matrix{Float64}(undef, n, n)
        XX = Vector{Float64}(undef, 2*n)
        nonstationary_variables = Vector{Bool}(undef, n)
        dgees_ws = DgeesWs(n)
        linsolve_ws1 = LinSolveWs(n)
        linsolve_ws2 = LinSolveWs(2*n)
        new(AA, AAtemp, AA2, BB, temp1, XX, nonstationary_variables,
            dgees_ws, linsolve_ws1, linsolve_ws2)
    end
end

function solve_one_row!(X, A, B, n, row, ws)
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
    linsolve_core!(vAA, vX, ws.linsolve_ws1)
    i = row
    j = n*(row - 1) + 1
    while i < row*n
        X[i] = X[j]
        i += n
        j += 1
    end
end

function solve_two_rows!(X, A, B, n, row, ws)
    α11, α21, α12, α22 = A[(row - 1):row, (row - 1):row]
    l1 = 1
    l2 = 1
    while l1 <= row
        k1 = 1
        k2 = 1
        while k1 <= row
            ws.AA2[k2, l2] = α11*A[k1, l1]
            ws.AA2[k2  + 1, l2] = α21*A[k1, l1]
            ws.AA2[k2, l2 + 1] = α12*A[k1, l1]
            ws.AA2[k2 + 1, l2 + 1] = α22*A[k1, l1]
            k1 += 1
            k2 += 2
        end
        ws.XX[l2] = -B[row - 1, l1]
        ws.XX[l2 + 1] = -B[row, l1]
        l1 += 1
        l2 += 2
    end
    vAA = view(ws.AA2, 1:2*row, 1:2*row)           
    vAA .-= I(2*row)
    vXX = view(ws.XX, 1:2*row)
    linsolve_core!(vAA, vXX, ws.linsolve_ws2)
    i = row - 1
    j = n*(row - 2) + 1
    m = 1
    while i <= row*n
        X[i] = vXX[m]
        X[i + 1] = vXX[m+1]
        X[j] = X[i]
        X[j + n] = X[i+1]
        i += n
        j += 1
        m += 2
    end
end

"""
    function lyapd(Σ, A, B, ws) solves equation Σ - A*Σ*A' = B
"""
function extended_lyapd!(Σ, A, B, ws)
    n = size(A, 1)
    copy!(ws.AA, A)
    dgees!(ws.dgees_ws, ws.AA, >, 1-1e-6)
    mul!(ws.temp1, transpose(ws.dgees_ws.vs), B)
    mul!(ws.BB, ws.temp1, ws.dgees_ws.vs)

    fill!(ws.nonstationary_variables, false)
    rowu = 1
    while rowu <= n
        if rowu == n || ws.AA[rowu + 1, rowu ] == 0
            if abs(ws.AA[rowu, rowu]) > 1-1e-6
                for i = 1:n
                    if abs(ws.dgees_ws.vs[i, rowu]) > 1e-10
                        ws.nonstationary_variables[i] = true
                    end
                end
            else
                break
            end
            rowu += 1
        else
            if ws.AA[rowu, rowu]*ws.AA[rowu,rowu] + ws.AA[rowu + 1, rowu]*ws.AA[rowu, rowu + 1] > 1-2e-6
                for i = 1:n
                    if abs(ws.dgees_ws.vs[i, rowu]) > 1e-10
                        ws.nonstationary_variables[i] = true
                    end
                end
                for i = 1:n
                    if abs(ws.dgees_ws.vs[i, rowu + 1]) > 1e-10
                        ws.nonstationary_variables[i] = true
                    end
                end
            else
                break
            end
            rowu += 2
                
        end
    end

    fill!(Σ, 0.0)
    row = n
    while row >= rowu
        if row == 1 || ws.AA[row, row - 1] == 0
            solve_one_row!(Σ, ws.AA, ws.BB, n, row, ws)
            vB = view(ws.BB, 1:row - 1, 1:row - 1)
            vtemp = view(ws.temp1, 1:row - 1, 1)
            vΣ = view(Σ, 1:row -1 , row)
            vA = view(ws.AA, 1:row - 1, 1:row - 1)
            mul!(vtemp, vA, vΣ)
            vA = view(ws.AA, 1:row - 1, row)
            mul!(vB, vtemp, vA', 1.0, 1.0)
            vΣ = view(Σ, 1:row, row)
            vA = view(ws.AA, 1:row - 1, 1:row)
            mul!(vtemp, vA, vΣ)
            vA = view(ws.AA, 1:row - 1, row)
            vB .= vB .+ vA.*vtemp'
            row -= 1
        else
            solve_two_rows!(Σ, ws.AA, ws.BB, n, row, ws)
            vB = view(ws.BB, 1:row - 2, 1:row - 2)
            vtemp = view(ws.temp1, 1:row - 2, 1:2)
            vΣ = view(Σ, 1:row - 2, row - 1:row)
            vA = view(ws.AA, 1:row - 2, 1:row - 2)
            mul!(vtemp, vA, vΣ)
            vA = view(ws.AA, 1:row - 2, row - 1:row)
            mul!(vB, vtemp, vA', 1.0, 1.0)
            vΣ = view(Σ, 1:row, row - 1:row)
            vA = view(ws.AA, 1:row - 2, 1:row)
            mul!(vtemp, vA, vΣ)
            vA = view(ws.AA, 1:row - 2, row - 1:row)
            mul!(vB, vA, vtemp', 1.0, 1.0)
            row -= 2
        end
    end

    mul!(ws.temp1, ws.dgees_ws.vs, Σ)
    mul!(Σ, ws.temp1, ws.dgees_ws.vs')

    if rowu > 1
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
end

