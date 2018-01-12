export indmax
"indmax(x, y, z)"
function Base.indmax(xs...) # mimic max(xs...)
    Int[indmax(collect(x)) for x in zip(xs...)]
end

"indmax(x, dim)"
function Base.indmax(x, dim::Int) # mimic maximum(x, dim)
    ind2sub(size(x), vec(findmax(x, dim)[2]))[dim]
end

export minimums
function minimums(x)
    mins = similar(x)
    xmin = x[1]
    @inbounds for i = eachindex(x)
        x[i] < xmin && (xmin = x[i])
        mins[i] = xmin
    end
    mins
end

export maximums
function maximums(x)
    maxs = similar(x)
    xmax = x[1]
    @inbounds for i = eachindex(x)
        x[i] > xmax && (xmax = x[i])
        maxs[i] = xmax
    end
    maxs
end

Base.max(itr) = maximum(itr)
Base.min(itr) = minimum(itr)

function Base.findmin(f::Function, x; parallel = false)
    y = parallel ? pmap(f, x) : map(f, x)
    yi, i = findmin(y)
    return x[i], y[i]
end

function Base.findmax(f::Function, x; parallel = false)
    y = parallel ? pmap(f, x) : map(f, x)
    yi, i = findmax(y)
    return x[i], y[i]
end
