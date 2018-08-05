export droplast
droplast(x, n = 1) = Iterators.take(x, length(x) - n)

function firstn(itr, n::Integer)
    a = Vector{eltype(itr)}(n)
    state = start(itr)
    for i in 1:n
        done(itr, state) && return a[1:i - 1]
        a[i], state = next(itr, state)
    end
    return a
end

firstn(a::AbstractArray, n::Integer) = a[1:min(end, n)]

lastn(a::AbstractArray, n::Integer) = a[max(1, end - n + 1):end]