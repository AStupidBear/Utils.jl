__precompile__(true)

module Utils

include("load.jl")
include("macros.jl")

include("array.jl")
include("ast.jl")
include("cmd.jl")
include("const.jl")
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
include("plot.jl")
include("rsync.jl")
include("statistics.jl")
include("string.jl")
include("sys.jl")
include("time.jl")
include("traits.jl")
include("sort.jl")
include("iterators.jl")
include("hash.jl")
include("mmap.jl")
include("fillna.jl")

include("encoder.jl")
include("scaler.jl")
include("data.jl")

export loadutil
loadutil(name) = include(joinpath(@__DIR__, name * ".jl"))

end # End of Utils
