Base.Int64(x::Union{Float64, Float32}) = round(Int64, x)
Base.Int32(x::Union{Float64, Float32}) = round(Int32, x)

export hasnan
function hasnan(x)
    for i in eachindex(x)
        isnan(x[i]) && return true
    end
    false
end
