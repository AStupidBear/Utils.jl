export sortall, sortall!

function sortall(xs::AbstractArray...; kw...)
    p = sortperm(first(xs); kw...)
    map(x -> x[p], xs)
end

function sortall!(xs::AbstractArray...; kw...)
    p = sortperm(first(xs); kw...)
    for x in xs
        permute!(x, p)
    end
    return xs
end