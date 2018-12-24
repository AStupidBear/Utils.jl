__precompile__(true)

module Utils

include("load.jl")
include("macros.jl")

include("array.jl")
include("ast.jl")
include("cmd.jl")
include("conv.jl")
include("download.jl")
include("function.jl")
include("io.jl")
include("math.jl")
include("maxmin.jl")
include("miscellaneous.jl")
include("number.jl")
include("parallel.jl")
include("parameter.jl")
include("statistics.jl")
include("sys.jl")
include("time.jl")
include("traits.jl")
include("sort.jl")
include("iterators.jl")
include("hash.jl")
include("mmap.jl")
include("fillna.jl")
include("string.jl")

function __init__()
    @require Atom="c52e3926-4ff0-5f6e-af25-54175e0327b1" include("atom.jl")
end

end # End of Utils
