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
                                        nonstationary_variables::Vector{Bool},
                                        state_nbr::Int64,
                                        nonstate_stationary_nbr)
        nonstate_nbr = endogenous_nbr - state_nbr
        state_stationary_variables = Vector{Bool}(undef, state_nbr)
        state_stationary_variables .= .!nonstationary_variables
        state_stationary_nbr = count(state_stationary_variables)
        rΣ_s_s = Matrix{Float64}(undef, state_stationary_nbr, state_stationary_nbr)
        rA1 = Matrix{Float64}(undef, state_stationary_nbr, state_stationary_nbr)
        rA2 = Matrix{Float64}(undef, nonstate_stationary_nbr, state_stationary_nbr)
        rB1 = Matrix{Float64}(undef, state_stationary_nbr, exogenous_nbr)
        rB2 = Matrix{Float64}(undef, nonstate_stationary_nbr, exogenous_nbr)
        rA2S = Matrix{Float64}(undef, nonstate_stationary_nbr, state_stationary_nbr)
        rB2S = Matrix{Float64}(undef, nonstate_stationary_nbr, exogenous_nbr)
        rΣ_ns_s = Matrix{Float64}(undef, nonstate_stationary_nbr, state_stationary_nbr)
        rΣ_ns_ns = Matrix{Float64}(undef, nonstate_stationary_nbr, nonstate_stationary_nbr)
        nonstate_stationary_variables = Vector{Bool}(undef, nonstate_nbr)
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
    state_stationary_variables::Vector{Bool}
    nonstate_stationary_variables::Vector{Bool}
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
            state_stationary_variables,
            nonstate_stationary_variables,
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
    @show state_indices
    @show nonstate_indices
    n = size(Σy, 1)
    fill!(Σy, NaN)
    m1 = m2 = 1
    n1 = n2 = 1
    for i = 1:n
        if i == state_indices[n1]
            if stationary_state_indices[n1]
                k1 = k2 = 1
                l1 = l2 = 1
                for j = 1:n
                    if j == state_indices[l1]
                        if state_stationary_indices[l1]
                            Σy[j,i] = Σ_s_s[k1, m1]
                            k1 += 1
                        end
                        l1 += 1
                    elseif nonstate_indices[l1]
                        Σy[j,i] = Σ_ns_s[k2, m1]
                        k2 += 1
                    end
                end
                m1 += 1
            end
            n1 += 1
        elseif nonstate_statonary_indices[n2]
            k1 = k2 = 1
            l1 = l2 = 1
            for j = 1:n
                if j == state_indices[l1]
                    Σy[j,i] = Σ_ns_s[m2, k1]
                    k1 += 1
                elseif nonstate_indices[l2]
                    Σy[j,i] = Σ_ns_ns[k2, m2]
                    k2 += 1
                end
                m2 += 1
            end
            n2 += 1
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
    @show ws.lyapd_ws.nonstationary_variables
    # stationary variables
    #=
    stationary_variables = ws.stationary_variables
    fill!(stationary_variables, true)
    state_stationary_variables =
        view(stationary_variables, lre_ws.backward_indices)
    nonstate_stationary_variables = view(stationary_variables,
                                         lre_ws.non_backward_indices)
    =#
    if is_stationary(ws.lyapd_ws)
        stationary_variance_blocks!(ws.Σ_ns_s, ws.Σ_ns_ns, A1, A2, B1,
                                    B2, ws.A2S, ws.B2S, ws.Σ_s_s, Σe)
        make_stationary_variance!(Σy, ws.Σ_s_s, ws.Σ_ns_s,
                                  ws.Σ_ns_ns, lre_ws.backward_indices)
    else
        nonstate_nbr = lre_ws.endogenous_nbr - lre_ws.backward_nbr 
        nonstate_stationary_variables = ws.nonstate_stationary_variables
        if length(nonstate_stationary_variables) != nonstate_nbr
            resize!(nonstate_stationary_variables, nonstate_nbr)
        end
        for i = 1:nonstate_nbr
            for j = nonstate_nbr
                if ws.lyapd_ws.nonstationary_variables[j] && abs(A2[i, j]) > 1e-10
                    nonstate_stationary_variables[j] = false
                    break
                end
            end
        end
        nonstate_stationary_nbr = count(nonstate_stationary_variables)
        if length(ws.nonstationary_ws) == 0
            nonstationary_ws = LRENonstationaryVarianceWs(lre_ws.endogenous_nbr,
                                                          lre_ws.exogenous_nbr,
                                                          ws.lyapd_ws.nonstationary_variables,
                                                          lre_ws.backward_nbr,
                                                          nonstate_stationary_nbr)
            push!(ws.nonstationary_ws, nonstationary_ws)
        else
            nonstationary_ws = ws.nonstationary_ws[1]
        end
        state_stationary_variables = nonstationary_ws.state_stationary_variables
        nonstationary_ws.rA1 .= view(A1, state_stationary_variables, state_stationary_variables)
        nonstationary_ws.rB1 .= view(B1, state_stationary_variables, :)
        if nonstate_stationary_nbr > 0
            nonstationary_ws.rA2 .= view(A2, nonstate_stationary_variables, :)
            nonstationary_ws.rB2 .= view(B2, nonstate_stationary_variables, :)
        end
        stationary_variance_blocks!(nonstationary_ws.rΣ_ns_s,
                                    nonstationary_ws.rΣ_ns_ns,
                                    nonstationary_ws.rA1,
                                    nonstationary_ws.rA2,
                                    nonstationary_ws.rB1,
                                    nonstationary_ws.rB2,
                                    nonstationary_ws.rA2S,
                                    nonstationary_ws.rB2S,
                                    nonstationary_ws.rΣ_s_s,
                                    Σe)
        make_nonstationary_variance!(Σy, ws.Σ_s_s, nonstationary_ws.rΣ_ns_s,
                                     nonstationary_ws.rΣ_ns_ns,
                                     lre_ws.backward_indices,
                                     nonstationary_ws.state_stationary_variables,
                                     lre_ws.non_backward_indices,
                                     nonstationary_ws.nonstate_stationary_variables)
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

#=
stationary_nbr = state_stationary_nbr + nonstate_stationary_nbr
    A2 = zeros(nonstate_stationary_nbr, state_stationary_nbr)
    vtmp = view(results.linearrationalexpectations.g1_1, m.i_non_states, :)
    A2 .= view(vtmp, nonstate_stationary_variables, state_stationary_variables)
    vtmp = view(Σy, m.i_bkwrd_b, m.i_bkwrd_b)
    vΣy = view(vtmp, state_stationary_variables, state_stationary_variables)
    vΣ = view(Σ, state_stationary_variables, state_stationary_variables) 
    vΣy .= vΣ
    # endogenous / nonstate
    n_non_states = m.endogenous_nbr - m.n_states
    B2 = zeros(nonstate_stationary_nbr, m.exogenous_nbr)
    vtmp = view(results.linearrationalexpectations.g1_2, m.i_non_states, :)
    vr2 = view(vtmp, nonstate_stationary_variables, :)
    B2 .= vr2
    Σ = zeros(stationary_nbr, nonstate_stationary_nbr)
    vg1 = view(results.linearrationalexpectations.g1_1, stationary_variables, state_stationary_variables)
    tmp1 = zeros(stationary_nbr, state_stationary_nbr)
    mul!(tmp1, vg1, vΣy)
    vg1 = view(results.linearrationalexpectations.g1_1, m.i_non_states, :)
    vg11 = view(vg1, nonstate_stationary_variables, state_stationary_variables)
    mul!(Σ, tmp1, transpose(vg11))
    tmp2 = zeros(stationary_nbr, m.exogenous_nbr)
    vg2 = view(results.linearrationalexpectations.g1_2, stationary_variables, :)
    mul!(tmp2, vg2, m.Sigma_e)
    mul!(Σ, tmp2, transpose(B2), 1.0, 1.0)
    vtmp = view(Σy, :, m.i_non_states) 
    vΣy = view(vtmp, stationary_variables, nonstate_stationary_variables)
    vΣy .= Σ
    # nonstate / state
    vtmp1 = view(Σy, m.i_non_states, m.i_bkwrd_b)
    vΣy1 = view(vtmp1, nonstate_stationary_variables, state_stationary_variables)
    vtmp2 = view(Σy, m.i_bkwrd_b, m.i_non_states)
    vΣy2 = view(vtmp2, state_stationary_variables, nonstate_stationary_variables)
    vΣy1 .= transpose(vΣy2)
end
=#    
