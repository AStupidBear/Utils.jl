export plus, minus
plus(x::Real) = ifelse(x > 0, one(x), zero(x))
minus(x::Real) = ifelse(x < 0, oftype(x, -1), zero(x))

export sign
Base.sign(x::Real, Θ) = ifelse(x < -Θ, oftype(x, -1), ifelse(x > Θ, one(x), zero(x)))

"pf = piecewise(:x,:([x>0, x==0, x<0]), :([2*x, -1, -x]))"
function piecewise(x::Symbol, c::Expr, f::Expr)
    n = length(f.args)
    @assert n == length(c.args)
    @assert c.head == :vect
    @assert f.head == :vect
    vf = Vector{Function}(n)
    for i in 1:n
        vf[i] = @eval $x->$(f.args[i])
    end
    return @eval ($x)->($(vf)[findfirst($c)])($x)
end

export rsqaured
rsqaured(ypred, y) = 1 - sum(abs2, ypred .- y) / sum(abs2, ypred .- mean(y))

grad1(x) = [mapslices(diff, x, dim) for dim in 1:ndims(x)]

grad2(x) = grad1.(grad1(x))

function area(coords)
    b = coords[end]
    v = 0.0
    for i in 1:length(coords)
        a, b = b, coords[i]
        v += a[2] * b[1] - a[1] * b[2]
    end
    return v * 0.5
end

function inbounds!(x, l, u)
    @inbounds for i in eachindex(x)
        if x[i] < l[i]
            x[i] = -x[i] + 2l[i]
        elseif x[i] > u[i]
            x[i] = -x[i] + 2u[i]
        end
    end
end

export ptp
function ptp(x)
    xmin, xmax = extrema(x)
    xmax - xmin
end

export diffn
function diffn(n, x, y = x)
    dx = zeros(x)
    for t in (n + 1):length(x)
        dx[t] = x[t] - y[t-n]
    end
    dx
end

export change
function change(x, T)
    dx = zeros(x)
    for t in linearindices(x)
        t′ = clamp(t - T, 1, length(x))
        t1, t2 = minmax(t, t′)
        dx[t] = x[t2] - x[t1]
    end
    dx
end

export pctchange
function pctchange(x, T)
    ϵ = eps(eltype(x))
    z = zeros(x)
    for t in linearindices(x)
        t′ = clamp(t - T, 1, length(x))
        t1, t2 = minmax(t, t′)
        z[t] = (x[t2] - x[t1]) / (x[t1] + ϵ)
    end
    return z
end

export cutoff
heaviside(x) = 0.5 * (1 + sign(x))
delta(x, δx = 1e-3) = (heaviside(x + δx / 2) - heaviside(x - δx / 2)) / δx
interval(x, xmin, xmax) = x > xmax ? 0.0 : (x < xmin ? 0.0 : 1.0)
cutoff{T}(x::T, xmin, xmax)::T =   x < xmin ? xmin : x > xmax ? xmax : x

export hardmax
softmax(x, dim = 1) = (y = exp.(x); y ./ sum(y, dim))
hardmax(x, dim = 1) = x .== maximum(x, dim)

Base.sum(f::Function, v0::Number, iter) = mapreduce(f, +, v0, iter)

Base.maximum(f::Function, v0::Number, iter) = isempty(iter) ? v0 : maximum(f, iter)

export ⧶
⧶(x, y) = x / y
⧶(x, y::AbstractFloat) = x / (y + eps(y))
⧶(x, y::Integer) = ifelse(x == y == 0, zero(x), x / y)