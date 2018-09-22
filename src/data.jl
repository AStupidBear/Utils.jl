export seq2stack
"""
x = rand(3,1000); y = rand(100,1); z = rand(100,3)
seq2stack(3, x); seq2stack(3, x, y);
seq2stack(3, x, y, z)
"""
function seq2stack(tstack::Int, seq::Array{T}) where T
    nr, nc = size(seq)
    stack = zeros(T, nr * tstack, nc)
    for t in 1:nc
        ind = ((t - tstack) * nr + 1):(t * nr)
        for i in 1:size(stack, 1)
            stack[i, t] = ind[i] > 0 ? seq[ind[i]] : seq[mod1(i, nr)]
        end
    end
    return stack
end

function seq2stack(tstack::Int, x::Array, ys...)
    xstack = seq2stack(tstack, x)
    ystack = []
    for y in ys
        push!(ystack, y[tstack:end, :])
    end
    xstack, ystack...
end

# function seqstack{T}(tstack::Int, x::Array{T})
#   stack = zeros(T, csize(x)..., tstack, ccount(x)-tstack+1)
#   for t = tstack:ccount(x)
#       cset!(stack, cget(x, t-tstack+1:t), t-tstack+1)
#   end
#   return stack
# end

export splitdata
function splitdata(x, y; splitratio = 0.2, featuredim = "col")
    if featuredim == "col"
        getfun, setfun, countfun = cview, cset!, ccount
    elseif featuredim == "row"
        getfun, setfun, countfun = rview, rset!, rcount
    end
    n = countfun(y)
    s = round(Int, (1 - splitratio) * n)
    xtrn, xtst = getfun(x, 1:s), getfun(x, (s + 1):n)
    ytrn, ytst = getfun(y, 1:s), getfun(y, (s + 1):n)
    return  xtrn, ytrn, xtst, ytst
end

export centralize, centralize!
"transform x to (-1, 1)"
function centralize!(x, dim = 1)
    𝑚ax = maximum(x, dim)
    𝑚in = minimum(x, dim)
    x .= (x .- 𝑚in) ./ (𝑚ax .- 𝑚in)
    x .= 2 .* x .- 1
end
centralize(x, dim = 1) = centralize!(deepcopy(x), dim)

Base.similar(x::Tuple) = deepcopy(x)
export balance
"""
```julia
using Utils
x = rand(4, 100)
y = randprob(-1:1, [0.2, 0.6, 0.2], 100)
xb, yb = balance(x, y)
hist1(y, -1.5:1.5)
hist1(yb, -1.5:1.5)
```
"""
function balance(x, y; featuredim = "col", sampleratio = 1.0)
    if featuredim == "col"
        getfun, setfun, countfun = cview, cset!, ccount
    elseif featuredim == "row"
        getfun, setfun, countfun = rview, rset!, rcount
    end

    d = Dict()
    for i in 1:ccount(x)
        get!(d, y[i], [getfun(x, i)])
        push!(d[y[i]], getfun(x, i))
    end

    nb = Int(countfun(x) * sampleratio); yb = zeros(nb)
    xb = featuredim == "col" ? tuple([zeros(size(z, 1), nb) for z in x]...) : tuple([zeros(nb, size(z, 2)) for z in x]...)

    ny, key, vals = length(d), collect(keys(d)), collect(values(d))
    for i in 1:nb
        r = rand(1:ny)
        setfun(yb, key[r], i)
        setfun(xb, rand(vals[r]), i)
    end
    return xb, yb
end

function Random.shuffle(x::Union{AbstractArray, Tuple}, y::Union{AbstractArray, Tuple}; featuredim = "col")
    if featuredim == "col"
        getfun, setfun, countfun = cview, cset!, ccount
    elseif featuredim == "row"
        getfun, setfun, countfun = rview, rset!, rcount
    end
    a = randperm(countfun(y))
    x, y = getfun(x, a), getfun(y, a)
end

export undersample
function undersample(xs::Array...; nsample = 2, featuredim = "col")
    if featuredim == "col"
        getfun, setfun, countfun = cview, cset!, ccount
    elseif featuredim == "row"
        getfun, setfun, countfun = rview, rset!, rcount
    end
    cols = rand(1:countfun(xs[1]), nsample)
    [getfun(x, cols) for x in xs]
end
