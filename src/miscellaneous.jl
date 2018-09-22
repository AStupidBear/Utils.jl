export @pygen
macro pygen(f)
    # generate a new symbol for the channel name and
    # wrapper function
    c = gensym()
    η = gensym()
    # yield(λ) → put!(c, λ)
    f′ = MacroTools.postwalk(f) do x
        @capture(x, yield(λ_)) || return x
        return :(put!($c, $λ))
    end
    # Fetch the function name and args
    @capture(f′, function func_(args__) body_ end)
    # wrap up the η function
    final = quote
        function $func($(args...))
            function $η($c)
                $body
            end
            return Channel(c -> $η(c))
        end
    end
    return esc(final)
end

export match
function Base.match(x, a::AbstractVector, b::AbstractVector)
    for i in eachindex(a)
        x == a[i] && return b[i]
    end
end

export pkill
function pkill(name)
    if iswindows()
        try run(pipeline(`pgrep $name`, `xargs -n 1 kill`)) finally end
    else
        try run(`pkill -f -9 $name`) finally end
    end
end

# function Base.edit(path::AbstractString, line::Integer=0)
#     command = Base.editor()
#     name = basename(first(command))
#     issrc = length(path)>2 && path[end-2:end] == ".jl"
#     if issrc
#         f = Base.find_source_file(path)
#         f !== nothing && (path = f)
#     end
#     cmd = line != 0 ? `$command -g $path:$line` : `$command $path`
#     spawn(pipeline(cmd, stderr=STDERR))
#     nothing
# end

# export smooth
# function smooth(x, n, dim::Integer = 1)
#     s = similar(x)
#     Rpre = CartesianRange(size(x)[1:dim-1])
#     Rpost = CartesianRange(size(x)[dim+1:end])
#     _smooth!(s, x, n, Rpre, size(x, dim), Rpost)
# end
#
# @noinline function _smooth!(s, x, n, Rpre, N, Rpost)
#     for Ipost in Rpost
#         for i = 1:N
#             ind = max(1, i-n):min(N, i+n)
#             for Ipre in Rpre
#                 s[Ipre, i, Ipost] = sum(x[Ipre, i, Ipost] for i in ind) / length(ind)
#             end
#         end
#     end
#     s
# end
