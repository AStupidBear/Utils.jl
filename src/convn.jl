using Base.Cartesian

export convn!, convn

convsize(w, x) = size(x) .- size(w) .+ 1

@generated function convn!(y::AbstractArray{T}, w::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    quote
        @inbounds begin
            @nloops $N j w begin
                @nloops $N k y begin
                    @nextract $N i d -> k_d + j_d - 1
                    (@nref $N y k) += (@nref $N x i) * (@nref $N w j)
                end
            end
        end
        return y
    end
end

@generated function convn(w::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    quote
        ysize = convsize(w, x)
        y = zeros(ysize)
        convn!(y, w, x)
        return y
    end
end

function conv4(w::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    W1, W2, I, O = size(w)
    X1, X2, I, N = size(x)
    y = zeros(T, (X1 - W1 + 1, X2 - W2 + 1, O, N))
    @inbounds for j in 1:N, i in 1:O
        convn!(view(y, :, :, i:i, j), view(w, :, :, :, i), view(x, :, :, :, j))
    end
    return y
end
