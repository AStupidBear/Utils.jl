export isna, fillna!

isna(x::Float32, θ = 0f0) = abs(x) < θ || isnan(x)

fillna(x, o...) = fillna!(copy(x), o...)

function fillna!(x::AbstractVector, o...)
    for t in 2:length(x)
        isna(x[t], o...) && (x[t] = x[t - 1])
    end
    for t in (length(x) - 1):-1:1
        isna(x[t], o...) && (x[t] = x[t + 1])
    end
    return x
end

function fillna!(x::AbstractMatrix, o...)
    for c in eachcol(x)
        fillna!(c, o...)
    end
    return x
end
