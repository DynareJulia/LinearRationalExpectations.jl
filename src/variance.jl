struct NonstationaryVarianceWs
    rΣ_s_s::Matrix{Float64}
    rA1::Matrix{Float64}
    rA2::Matrix{Float64}
    rB1::Matrix{Float64}
    rB2::Matrix{Float64}
    rA2S::Matrix{Float64}
    rB2S::Matrix{Float64}
    rΣ_ns_s::Matrix{Float64}
    rΣ_ns_ns::Matrix{Float64}
    state_stationary_variables::Vector{Bool}
    nonstate_stationary_variables::Vector{Bool}
    function NonstationaryVarianceWs(endogenous_nbr::Int,
                                        exogenous_nbr::Int,
                                        state_nbr::Int,
                                        nonstationary_variables::Vector{Bool},
                                        A2::Matrix{Float64})
        nonstate_nbr = endogenous_nbr - state_nbr
        state_stationary_variables = Vector{Bool}(undef, state_nbr)
        nonstate_stationary_variables = Vector{Bool}(undef, nonstate_nbr)
        state_stationary_variables .= .!nonstationary_variables
        state_stationary_nbr = count(state_stationary_variables)
        fill!(nonstate_stationary_variables, true)
        for i = 1:nonstate_nbr
            for j = 1:state_nbr
                if nonstationary_variables[j] && abs(A2[i, j]) > 1e-10
                    nonstate_stationary_variables[i] = false
                    break
                end
            end
        end
        nonstate_stationary_nbr = count(nonstate_stationary_variables)
        rΣ_s_s = Matrix{Float64}(undef, state_stationary_nbr, state_stationary_nbr)
        rA1 = Matrix{Float64}(undef, state_stationary_nbr, state_stationary_nbr)
        rA2 = Matrix{Float64}(undef, nonstate_stationary_nbr, state_stationary_nbr)
        rB1 = Matrix{Float64}(undef, state_stationary_nbr, exogenous_nbr)
        rB2 = Matrix{Float64}(undef, nonstate_stationary_nbr, exogenous_nbr)
        rA2S = Matrix{Float64}(undef, nonstate_stationary_nbr, state_stationary_nbr)
        rB2S = Matrix{Float64}(undef, nonstate_stationary_nbr, exogenous_nbr)
        rΣ_ns_s = Matrix{Float64}(undef, nonstate_stationary_nbr, state_stationary_nbr)
        rΣ_ns_ns = Matrix{Float64}(undef, nonstate_stationary_nbr, nonstate_stationary_nbr)
        new(rΣ_s_s, rA1, rA2, rB1, rB2,
            rA2S, rB2S, rΣ_ns_s, rΣ_ns_ns,
            state_stationary_variables,
            nonstate_stationary_variables)
    end
end

struct VarianceWs
    B1S::Matrix{Float64}
    B1SB1::Matrix{Float64}
    A2S::Matrix{Float64}
    B2S::Matrix{Float64}
    Σ_s_s::Matrix{Float64}
    Σ_ns_s::Matrix{Float64}
    Σ_ns_ns::Matrix{Float64}
    stationary_variables::Vector{Bool}
    nonstationary_ws::Vector{NonstationaryVarianceWs}
    lre_ws::LinearRationalExpectationsWs
    lyapd_ws::LyapdWs
end
function VarianceWs(var_nbr::Int, lyapd_ws::LyapdWs,
                    shock_nbr::Int, lre_ws::LinearRationalExpectationsWs)
    state_nbr = size(lyapd_ws.AA, 1)
    nonstate_nbr = var_nbr - state_nbr
    B1S = Matrix{Float64}(undef, state_nbr, shock_nbr)
    B1SB1 = Matrix{Float64}(undef, state_nbr, state_nbr)
    A2S = Matrix{Float64}(undef, nonstate_nbr, state_nbr)
    B2S = Matrix{Float64}(undef, nonstate_nbr, shock_nbr)
    Σ_s_s = Matrix{Float64}(undef, state_nbr, state_nbr)
    Σ_ns_s = Matrix{Float64}(undef, nonstate_nbr, state_nbr)
    Σ_ns_ns = Matrix{Float64}(undef, nonstate_nbr, nonstate_nbr)
    stationary_variables = Vector{Bool}(undef, var_nbr)
    nonstationary_ws = Vector{NonstationaryVarianceWs}(undef, 0)
    VarianceWs(B1S, B1SB1, A2S, B2S, Σ_s_s, Σ_ns_s, Σ_ns_ns,
        stationary_variables,
        nonstationary_ws,
        lre_ws, lyapd_ws)
end
VarianceWs(var_nbr::Int, state_nbr::Int, shock_nbr::Int, lre_ws::LinearRationalExpectationsWs) = 
    VarianceWs(var_nbr, LyapdWs(state_nbr), shock_nbr, lre_ws)

function make_stationary_variance!(Σy::Matrix{Float64},
                                   Σ_s_s::Matrix{Float64},
                                   Σ_ns_s::Matrix{Float64},
                                   Σ_ns_ns::Matrix{Float64},
                                   backward_indices)
    n = size(Σy, 1)
    nb = length(backward_indices)
    m1 = m2 = 1
    @inbounds for i = 1:n
        if m1 <= nb && i == backward_indices[m1]
            k1 = k2 = 1
            for j = 1:n
                if k1 <= nb && j == backward_indices[k1]
                    Σy[j,i] = Σ_s_s[k1, m1]
                    k1 += 1
                else
                    Σy[j,i] = Σ_ns_s[k2, m1]
                    k2 += 1
                end
            end
            m1 += 1
        else
            k1 = k2 = 1
            for j = 1:n
                if k1 <= nb && j == backward_indices[k1]
                    Σy[j,i] = Σ_ns_s[m2, k1]
                    k1 += 1
                else
                    Σy[j,i] = Σ_ns_ns[k2, m2]
                    k2 += 1
                end
            end
            m2 += 1
            
        end
    end
end

function make_nonstationary_variance!(Σy::Matrix{Float64},
                                      Σ_s_s::Matrix{Float64},
                                      Σ_ns_s::Matrix{Float64},
                                      Σ_ns_ns::Matrix{Float64},
                                      state_indices,
                                      state_stationary_indices,
                                      nonstate_indices,
                                      nonstate_stationary_indices)
    n = size(Σy, 1)
    fill!(Σy, NaN)
    m1 = 1
    for (m2, i) in enumerate(state_indices)
        if state_stationary_indices[m2]
            k1 = k2 = 1
            for j in state_indices
                if state_stationary_indices[k2]
                    Σy[j,i] = Σ_s_s[k1, m1]
                    k1 += 1
                end
                k2 += 1
            end
            k1 = k2 = 1
            for j in nonstate_indices
                if nonstate_stationary_indices[k2]
                    Σy[j,i] = Σ_ns_s[k1, m1]
                    k1 += 1
                end
                k2 += 1
            end
            m1 += 1
        end
    end
    m1 = 1
    for (m2, i) in enumerate(nonstate_indices)
        if nonstate_stationary_indices[m2]
            k1 = k2 = 1
            for j in state_indices
                if state_stationary_indices[k2]
                    Σy[j,i] = Σ_ns_s[m1, k1]
                    k1 += 1
                end
                k2 += 1
            end
            k1 = k2 = 1
            for j in nonstate_indices
                if nonstate_stationary_indices[k2]
                    Σy[j,i] = Σ_ns_ns[k1, m1]
                    k1 += 1
                end
                k2 += 1
            end
            m1 += 1
        end
    end
end

function compute_variance!(Σy::Matrix{Float64},
                           A1::AbstractVecOrMat{Float64},
                           A2::AbstractVecOrMat{Float64},
                           B1::AbstractVecOrMat{Float64},
                           B2::AbstractVecOrMat{Float64},
                           Σe::AbstractVecOrMat{Float64},
                           ws::VarianceWs)
    n = size(Σy, 1)
    lre_ws = ws.lre_ws
    mul!(ws.B1S, B1, Σe)
    mul!(ws.B1SB1, ws.B1S, transpose(B1))
    # state variables variance
    extended_lyapd!(ws.Σ_s_s, A1, ws.B1SB1, ws.lyapd_ws)
    if is_stationary(ws.lyapd_ws)
        stationary_variance_blocks!(ws.Σ_ns_s, ws.Σ_ns_ns, A1, A2, B1,
                                    B2, ws.A2S, ws.B2S, ws.Σ_s_s, Σe)
        make_stationary_variance!(Σy, ws.Σ_s_s, ws.Σ_ns_s,
                                  ws.Σ_ns_ns, lre_ws.ids.backward)
        fill!(ws.stationary_variables, true)
    else
        state_nbr = n_backward(lre_ws.ids)
        state_indices = lre_ws.ids.backward
        if length(ws.nonstationary_ws) == 0
            nonstationary_ws = NonstationaryVarianceWs(n_endogenous(lre_ws.ids),
                                                       n_exogenous(lre_ws.ids),
                                                       n_backward(lre_ws.ids),
                                                       ws.lyapd_ws.nonstationary_variables,
                                                       A2)
            push!(ws.nonstationary_ws, nonstationary_ws)
        else
            nonstationary_ws = ws.nonstationary_ws[1]
        end
        state_stationary_variables = nonstationary_ws.state_stationary_variables
        nonstate_stationary_variables = nonstationary_ws.nonstate_stationary_variables
        fill!(ws.stationary_variables, false)
        m1 = m2 = 1
        for i = 1:size(Σy, 1)
            if m1 <= state_nbr && i == state_indices[m1]
                if state_stationary_variables[m1]
                    ws.stationary_variables[i] = true
                end
                m1 += 1
            else
                if nonstate_stationary_variables[m2]
                    ws.stationary_variables[i] = true
                end
                m2 += 1
            end
        end
        rΣ_ns_s = nonstationary_ws.rΣ_ns_s
        rΣ_ns_ns = nonstationary_ws.rΣ_ns_ns
        rA1    = nonstationary_ws.rA1   
        rA2    = nonstationary_ws.rA2   
        rB1    = nonstationary_ws.rB1   
        rB2    = nonstationary_ws.rB2   
        rA2S   = nonstationary_ws.rA2S  
        rB2S   = nonstationary_ws.rB2S  
        rΣ_s_s = nonstationary_ws.rΣ_s_s
        rΣ_s_s .= view(ws.Σ_s_s, state_stationary_variables, state_stationary_variables)
        rA1 .= view(A1, state_stationary_variables, state_stationary_variables)
        rB1 .= view(B1, state_stationary_variables, :)
        if any(nonstate_stationary_variables)
            rA2 .= view(A2, nonstate_stationary_variables, state_stationary_variables)
            rB2 .= view(B2, nonstate_stationary_variables, :)
        end
        stationary_variance_blocks!(rΣ_ns_s,
                                    rΣ_ns_ns,
                                    rA1,   
                                    rA2,   
                                    rB1,   
                                    rB2,   
                                    rA2S,  
                                    rB2S,  
                                    rΣ_s_s,
                                    Σe)
        make_nonstationary_variance!(Σy,
                                     rΣ_s_s,
                                     rΣ_ns_s,
                                     rΣ_ns_ns,
                                     lre_ws.ids.backward,
                                     state_stationary_variables,
                                     lre_ws.ids.non_backward,
                                     nonstate_stationary_variables)
    end
end

function compute_variance!(
    lreresults::LinearRationalExpectationsResults,
    Σe::AbstractVecOrMat{Float64},
    ws::VarianceWs
)
    A1 = lreresults.gs1
    A2 = lreresults.gns1
    B1 = lreresults.hs1
    B2 = lreresults.hns1
    endogenous_variance = compute_variance!(lreresults.endogenous_variance, A1, A2, B1, B2, Σe, ws)
    empty!(lreresults.stationary_variables)
    append!(lreresults.stationary_variables, ws.stationary_variables)
    return endogenous_variance
end

function compute_variance!(
    variance::AbstractMatrix{Float64},
    lreresults::LinearRationalExpectationsResults,
    Σe::AbstractVecOrMat{Float64},
    ws::VarianceWs
)
    A1 = lreresults.gs1
    A2 = lreresults.gns1
    B1 = lreresults.hs1
    B2 = lreresults.hns1
    endogenous_variance = compute_variance!(variance, A1, A2, B1, B2, Σe, ws)
    return endogenous_variance
end

function stationary_variance_blocks!(Σ_ns_s, Σ_ns_ns, A1, A2, B1, B2, A2S, B2S, Σ_s_s, Σe)
    mul!(B2S, B2, Σe)
    mul!(Σ_ns_s, B2S, transpose(B1))
    mul!(Σ_ns_ns, B2S, transpose(B2))
    mul!(A2S, A2, Σ_s_s)
    mul!(Σ_ns_s, A2S, transpose(A1), 1.0, 1.0)
    mul!(Σ_ns_ns, A2S, transpose(A2), 1.0, 1.0)
end

"""
    correlation!(c::AbstractMatrix{T}, v::AbstractMatrix{T}, sd::AbstractVector{T}) where T <: Real
computes the correlation coefficients c[i,j] = v[i,j]/(sd[i]*sd[j])
"""
function correlation!(c::AbstractMatrix{T}, v::AbstractMatrix{T}, sd::AbstractVector{T}) where T <: Real
    c .= v ./ (sd .* transpose(sd))
end

robustsqrt(x) = sqrt(x + eps())
"""
    correlation(v::AbstractMatrix{T}) where T <: Real
returns the correlation coefficients v[i,j]/sqrt(v[i,i]*v[j,j])
"""
function correlation(v::AbstractMatrix{T}) where T
    sd = robustsqrt.(diag(v))
    c = similar(v)
    correlation!(c, v, sd)
end

"""
autocovariance!(av::Vector{<:AbstractMatrix{T}}, a::AbstractMatrix{T}, v::AbstractMatrix{T}, work1::AbstractMatrix{T}, work2::AbstractMatrix{T},order::Int)

returns a vector of autocovariance matrices E(y_t y_{t-i}') i = 1,...,i for an vector autoregressive process y_t = Ay_{t-1} + Be_t
"""
function autocovariance!(av::Vector{<:AbstractMatrix{T}},
                         lre_results::LinearRationalExpectationsResults,
                         S1a::AbstractMatrix{T},
                         S1b::AbstractMatrix{T},
                         S2::AbstractMatrix{T},
                         backward_indices::Vector{Int},
                         stationary_variables::Vector{Bool}) where T <: Real
    S1a .= view(lre_results.endogenous_variance, backward_indices, :)
    n = size(av[1], 1)
    nb = length(backward_indices)
    for p in 1:length(av)
        mul!(S1b, lre_results.gs1, S1a)
        mul!(S2, lre_results.gns1, S1a)
        @inbounds for i = 1:n
            k1 = k2 = 1
            for j = 1:n
                if k1 <= nb && j == backward_indices[k1]
                    av[p][j,i] = S1b[k1, i]
                    k1 += 1
                else
                    av[p][j,i] = S2[k2, i]
                    k2 += 1
                end
            end
        end
        tmp = S1a
        S1a = S1b
        S1b = tmp
    end
    return av
end
    
"""
autocovariance!(av::Vector{<:AbstractVector{T}}, a::AbstractMatrix{T}, v::AbstractMatrix{T}, work1::AbstractMatrix{T}, work2::AbstractMatrix{T},order::Int)

returns a vector of autocovariance vector with elements E(y_{j,t} y_{j,t-i}') j= 1,...,n and i = 1,...,i for an vector autoregressive process y_t = Ay_{t-1} + Be_t
"""
function autocovariance!(av::Vector{<:AbstractVector{T}},
                         lre_results::LinearRationalExpectationsResults,
                         S1a::AbstractMatrix{T},
                         S1b::AbstractMatrix{T},
                         S2::AbstractMatrix{T},
                         backward_indices::Vector{Int},
                         stationary_variables::Vector{Bool}) where T <: Real
    backward_stationary_indices = [i for i in backward_indices if stationary_variables[i]]
    variance = lre_results.endogenous_variance
    nbsi = length(backward_stationary_indices)
    vS1a = view(S1a, 1:nbsi, :)
    vS1b = view(S1b, 1:nbsi, :)
    vS1a .= view(variance, backward_stationary_indices, :)
    n = length(av[1])
    nb = length(backward_indices)
    for i in 1:length(av)
        mul!(vS1b, lre_results.gs1, vS1a)
        mul!(S2, lre_results.gns1, vS1a)
        k1 = k2 = 1
        for j = 1:n
            if k1 <= nb && j == backward_indices[k1]
                av[i][j] = vS1b[k1, j]
                k1 += 1
            else
                av[i][j] = S2[k2, j]
                k2 += 1
            end
        end
        vS1a, vS1b = vS1b, vS1a
    end
    return av
end

function autocorrelation!(ar::Vector{<:AbstractVecOrMat{T}},
                          lre_results::LinearRationalExpectationsResults,
                          S1a::AbstractMatrix{T},
                          S1b::AbstractMatrix{T},
                          S2::AbstractMatrix{T},
                          backward_indices::Vector{Int},
                          stationary_variables::Vector{Bool}
                          ) where T <: Real
    autocovariance!(ar, lre_results, S1a, S1b, S2, backward_indices, stationary_variables)
    n = length(stationary_variables)
    dv = Vector{Float64}(undef, n)
    for i = 1:n
        dv[i] = lre_results.endogenous_variance[i, i]
    end
    autocorrelation!(ar, dv) 
end

"""
    autocorrelation!(ar::Vector{<:AbstractMatrix{T}}. dv::AbstractVector{T}) where T
returns a vector of autocorrelation matrices corresponging to the autocovariance matrices av
"""
function autocorrelation!(ar::Vector{<:AbstractMatrix{T}}, dv::AbstractVector{T}) where T
    n, m = size(ar[1], 1)
    for a in ar
        for i = 1:n
            for j = 1:m
                a[j, i] ./= sqrt(dv[i]*dv[j])
            end
        end
    end
    return ar
end
        
"""
    autocorrelation!(ar::Vector{<:AbstractVector{T}}, dv::AbstractVector{T}) where T
returns a vector of autocorrelation vectors corresponging to the autocovariance vector av
"""
function autocorrelation!(ar::Vector{<:AbstractVector{T}}, dv::AbstractVector{T}) where T
    for a in ar
        a ./= dv
    end
    return ar
end
        
"""
    autocorrelation!(ar::Vector{<:AbstractMatrix{T}}, av::Vector{<:AbstractMatrix{T}}, v::AbstractMatrix{T}) where T
returns a vector of autocorrelation matrices corresponging to the autocovariance matrices av
"""
function autocorrelation!(ar::Vector{<:AbstractMatrix{T}}, av::Vector{<:AbstractMatrix{T}}, v::AbstractMatrix{T}) where T
    n = size(v, 1)
    dv = diag(v)
    for (a1, a2) in zip(ar, av)
        for i = 1:n
            for j = 1:m
                a1[i,j] .= a2[i, j] ./ sqrt(dv[i]*dv[j])
            end
        end
    end
    return ar
end
        
"""
    autocorrelation!(ar::Vector{<:AbstractVector{T}}, av::Vector{<:AbstractMatrix{T}}, v::AbstractMatrix{T}) where T
returns a vector of autocorrelation vectors corresponging to the autocovariance vector av
"""
function autocorrelation!(ar::Vector{<:AbstractVector{T}}, av::Vector{<:AbstractMatrix{T}}, v::AbstractMatrix{T}) where T
    for (a1, a2) in zip(ar, av)
        a1 .= a2 ./ diag(v)
    end
    return ar
end
        
function variance_decomposition!(
    VD::AbstractMatrix{<:T},
    LRE_results::LinearRationalExpectationsResults,
    Σe::AbstractMatrix{<:T},
    variance::AbstractVector{<:T},
    work1::AbstractMatrix{<:T},
    work2::AbstractMatrix{<:T},
    lre_variance_ws::LinearRationalExpectations.VarianceWs) where T
    
    nx = size(Σe, 1)
    # force Σe to be positive definite
    cs = cholesky(Σe + 1e-14*I).L
    for i = 1:nx
        fill!(work1, 0.0)
        # use the vector for ith exogenous variable
        vcs = view(cs, :, i)
        work1 .= vcs*transpose(vcs)
        compute_variance!(work2,
                          LRE_results,
                          work1,
                          lre_variance_ws,
                          )
        for j = 1:size(VD, 1)
            VD[j, i] = work2[j, j]/variance[j]
        end
    end
    return VD
end

function variance_decomposition(LREresults::LinearRationalExpectationsResults,
                                LREws::LinearRationalExpectationsWs,
                                Σe::Matrix{Float64},
                                endogenous_nbr::Int,
                                exogenous_nbr::Int,
                                state_nbr::Int)
    VD = Matrix{Float64}(undef, endogenous_nbr, exogenous_nbr)
    work1 = Matrix{Float64}(undef, exogenous_nbr, exogenous_nbr)
    work2 = Matrix{Float64}(undef, endogenous_nbr, endogenous_nbr)
    LREvariance_ws = VarianceWs(
        endogenous_nbr,
        state_nbr,
        exogenous_nbr,
        LREws,
    )
    VD = variance_decomposition!(
        VD,
        LREresults,
        Σe,
        diag(LREresults.endogenous_variance),
        work1,
        work2,
        LREvariance_ws
    )
    return VD
end
                                 
