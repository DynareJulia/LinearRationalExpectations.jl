struct LREVarianceWs
    tmp_nst_nsh::Matrix{Float64}
    LREws::LREWs
    function LREVarianceWs(nvar::Int64, nstates::Int64, nshocks::Int64, LREws::LREWs)
        tmp_nst_nsh = zeros(LREws.backward_nbr + LREws.both_nbr, LREws.exogenous_nbr)
        new(tmp_nst_nsh, LREws)
end

"""
    mul3!(D, A, B, C, tmp)

computes D = A*B*C

using workspace tmp
"""

function mul3!(D::AbstractVecOrMat{Float64},
               A::AbstractVecOrMat{Float64},
               B::AbstractVecOrMat{Float64},
               C::AbstractVecOrMat{Float64},
               tmp::AbstractVecOrMat{Float64})
    mul!(tmp, B, C)
    mul!(D, A, tmp)
end
    
function compute_variance!(Σy::Matrix{Float64},
                           A::AbstractVecOrMat{Float64},
                           B::AbstractVecOrMat{Float64},
                           Σe::AbstractVecOrMat{Float64},
                           ws::LREVarianceWs)

    A = results.linearrationalexpectations.gs1
    B1 = zeros(m.n_states, m.exogenous_nbr)
    g1_1 = results.linearrationalexpectations.g1_1
    g1_2 = results.linearrationalexpectations.g1_2
    vr1 = view(g1_2, m.i_bkwrd_b, :)
    B1 .= vr1
    ws = LyapdWs(m.n_states::Int64)
    Σ = zeros(m.n_states, m.n_states)
    tmp = zeros(m.n_states, m.exogenous_nbr)
    mul!(tmp, B1, m.Sigma_e)
    B = zeros(m.n_states, m.n_states)
    mul!(B, tmp, B1')
    extended_lyapd!(Σ, A, B, ws)
    stationary_variables = results.stationary_variables
    fill!(stationary_variables, true)
    state_stationary_variables =
    view(stationary_variables, m.i_bkwrd_b)
    nonstate_stationary_variables = view(stationary_variables,
                                         m.i_non_states)
    if is_stationary(ws)
        state_stationary_nbr = m.n_states
        nonstate_stationary_nbr = m.endogenous_nbr - m.n_states
    else
        fill!(Σy, NaN)
        state_stationary_variables .= .!ws.nonstationary_variables
        state_stationary_nbr = count(state_stationary_variables)
        vr3 = view(results.linearrationalexpectations.g1_1, m.i_non_states, :)
        for i = 1:(m.endogenous_nbr - m.n_states)
            for j = 1:m.n_states
                if ws.nonstationary_variables[j] && abs(vr3[i, j]) > 1e-10
                    nonstate_stationary_variables[j] = false
                    break
                end
            end
        end
        nonstate_stationary_nbr = count(nonstate_stationary_variables)
    end
    # state / state
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
    
