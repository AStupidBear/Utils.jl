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
