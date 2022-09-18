function fan_columns!(y::Matrix{Float64}, x::Matrix{Float64}, columns::Vector{Int},
                      offset_x::Int)
    my, ny = size(y)
    mx, nx = size(x)
    source = offset_x*mx + 1
    @inbounds for i = 1:length(columns)
        k = columns[i]
        destination = (k - 1)*my + 1
        copyto!(y, destination, x, source, mx)
        source += mx
    end
end

function get_abc!(a::Matrix{Float64}, b::Matrix, c::Matrix{Float64}, jacobian::Matrix{Float64}, m::Model)
    fill!(a, 0.0)
    fill!(b, 0.0)
    fill!(c, 0.0)
    i_bkwrd = m.i_bkwrd_b
    i_current = m.i_current
    i_fwrd = m.i_fwrd_b
    fan_columns!(a, jacobian, m.i_bkwrd_b, 0)
    offset = m.n_bkwrd + m.n_both
    fan_columns!(b, jacobian, m.i_current, offset)
    offset += m.n_current
    fan_columns!(c, jacobian, m.i_fwrd_b, offset)
end

"""
    function h0!(h0::Matrix{Float64}, a::Matrix{Float64}, b::Matrix{Float64}, c::Matrix{Float64})

computes h0 = inv(a+b*c)
"""
function h0!(h0::AbstractMatrix{Float64}, a::AbstractMatrix{Float64},
             b::AbstractMatrix{Float64}, c::AbstractMatrix{Float64},
             work::AbstractMatrix{Float64}, linsolve_ws)
    @inbounds copy!(work, a)
    @inbounds mul!(work, b, c, 1.0, 1.0)
    fill!(h0, 0.0)
    n = size(h0, 1)
    m = 1
    @inbounds for i = 1:n
        h0[m] = 1.0
        m += n + 1
    end
    lu_t = LU(factorize!(linsolve_ws, work)...)
    ldiv!(lu_t, h0)
    if any(isnan.(h0))
        if any(isnan.(a))
            throw(ArgumentError("NaN in a"))
        end
        if any(isnan.(b))
            throw(ArgumentError("NaN in b"))
        end
        if any(isnan.(c))
            throw(ArgumentError("NaN in c"))
        end
        throw(ArgumentError("NaN in h0"))
    end
end

"""
function hh!(hh::AbstractMatrix{Float64}, h::AbstractMatrix{Float64}, 
             f::AbstractMatrix{Float64}, hf::AbstractMatrix{Float64},
             n::Int, work1::AbstractMatrix{Float64},
             work2::AbstractMatrix{Float64})

computes hh = [h, h(1), h(2), ..., h(n-1)] and h(i) = -h*f*h(-1)
"""
function hh!(hh::AbstractMatrix{Float64}, h::AbstractMatrix{Float64},
             f::AbstractMatrix{Float64}, hf::AbstractMatrix{Float64},
             preconditioner_window::Int, work1::AbstractMatrix{Float64},
             work2::AbstractMatrix{Float64})
    if any(isnan.(h))
        throw(ArgumentError("NaN in h"))
    end
    if any(isnan.(f))
        throw(ArgumentError("NaN in f"))
    end
    m = size(h, 1)
    m2 = m*m
    @inbounds mul!(hf, h, f, -1, 0)
    @inbounds copy!(work1, h)
    k = 1
    @inbounds for i = 1:preconditioner_window
        copyto!(hh, k, work1, 1, m2)
        if i < preconditioner_window
            mul!(work2, hf, work1)
            copy!(work1, work2)
            k += m2
        end
    end
end

