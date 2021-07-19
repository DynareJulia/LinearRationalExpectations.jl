struct LRENonstationaryVarianceWs
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
    function LRENonstationaryVarianceWs(endogenous_nbr::Int64,
                                        exogenous_nbr::Int64,
                                        state_nbr::Int64,
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

struct LREVarianceWs
    B1S::Matrix{Float64}
    B1SB1::Matrix{Float64}
    A2S::Matrix{Float64}
    B2S::Matrix{Float64}
    Σ_s_s::Matrix{Float64}
    Σ_ns_s::Matrix{Float64}
    Σ_ns_ns::Matrix{Float64}
    stationary_variables::Vector{Bool}
    nonstationary_ws::Vector{LRENonstationaryVarianceWs}
    lre_ws::LinearRationalExpectationsWs
    lyapd_ws::LyapdWs
    function LREVarianceWs(var_nbr::Int64, state_nbr::Int64,
                           shock_nbr::Int64, lre_ws::LinearRationalExpectationsWs)
        nonstate_nbr = var_nbr - state_nbr
        B1S = Matrix{Float64}(undef, state_nbr, shock_nbr)
        B1SB1 = Matrix{Float64}(undef, state_nbr, state_nbr)
        A2S = Matrix{Float64}(undef, nonstate_nbr, state_nbr)
        B2S = Matrix{Float64}(undef, nonstate_nbr, shock_nbr)
        Σ_s_s = Matrix{Float64}(undef, state_nbr, state_nbr)
        Σ_ns_s = Matrix{Float64}(undef, nonstate_nbr, state_nbr)
        Σ_ns_ns = Matrix{Float64}(undef, nonstate_nbr, nonstate_nbr)
        stationary_variables = Vector{Bool}(undef, var_nbr)
        state_stationary_variables = Vector{Bool}(undef, 0)
        nonstate_stationary_variables = Vector{Bool}(undef, 0)
        nonstationary_ws = Vector{LRENonstationaryVarianceWs}(undef, 0)
        lyapd_ws = LyapdWs(state_nbr)
        new(B1S, B1SB1, A2S, B2S, Σ_s_s, Σ_ns_s, Σ_ns_ns,
            stationary_variables,
            nonstationary_ws,
            lre_ws, lyapd_ws)
    end
end

function make_stationary_variance!(Σy::Matrix{Float64},
                                   Σ_s_s::Matrix{Float64},
                                   Σ_ns_s::Matrix{Float64},
                                   Σ_ns_ns::Matrix{Float64},
                                   backward_indices)
    n = size(Σy, 1)
    m1 = m2 = 1
    @inbounds for i = 1:n
        if i == backward_indices[m1]
            k1 = k2 = 1
            for j = 1:n
                if j == backward_indices[k1]
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
                if j == backward_indices[k1]
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
    m1 = m2 = 1
    for i in state_indices
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
        m2 += 1
    end
    m = 1
    for i in nonstate_indices
        if nonstate_stationary_indices[m]
            k1 = k2 = 1
            for j in state_indices
                if state_stationary_indices[k2]
                    Σy[j,i] = Σ_ns_s[m, k1]
                    k1 += 1
                end
                k2 += 1
            end
            k1 = k2 = 1
            for j in nonstate_indices
                if nonstate_stationary_indices[k2]
                    Σy[j,i] = Σ_ns_ns[k1, m]
                    k1 += 1
                end
                k2 += 1
            end
            m += 1
        end
    end
end

function compute_variance!(Σy::Matrix{Float64},
                           A1::AbstractVecOrMat{Float64},
                           A2::AbstractVecOrMat{Float64},
                           B1::AbstractVecOrMat{Float64},
                           B2::AbstractVecOrMat{Float64},
                           Σe::AbstractVecOrMat{Float64},
                           ws::LREVarianceWs)
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
                                  ws.Σ_ns_ns, lre_ws.backward_indices)
    else
        if length(ws.nonstationary_ws) == 0
            nonstationary_ws = LRENonstationaryVarianceWs(lre_ws.endogenous_nbr,
                                                          lre_ws.exogenous_nbr,
                                                          lre_ws.backward_nbr,
                                                          ws.lyapd_ws.nonstationary_variables,
                                                          A2)
            push!(ws.nonstationary_ws, nonstationary_ws)
        else
            nonstationary_ws = ws.nonstationary_ws[1]
        end
        state_stationary_variables = nonstationary_ws.state_stationary_variables
        nonstate_stationary_variables = nonstationary_ws.nonstate_stationary_variables
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
                                     lre_ws.backward_indices,
                                     state_stationary_variables,
                                     lre_ws.non_backward_indices,
                                     nonstate_stationary_variables)
    end
end
    
function stationary_variance_blocks!(Σ_ns_s, Σ_ns_ns, A1, A2, B1, B2, A2S, B2S, Σ_s_s, Σe)
    mul!(B2S, B2, Σe)
    mul!(Σ_ns_s, B2S, transpose(B1))
    mul!(Σ_ns_ns, B2S, transpose(B2))
    mul!(A2S, A2, Σ_s_s)
    mul!(Σ_ns_s, A2S, transpose(A1), 1.0, 1.0)
    mul!(Σ_ns_ns, A2S, transpose(A2), 1.0, 1.0)
end
