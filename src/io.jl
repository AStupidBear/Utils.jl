export readabsdir
readabsdir(dir) = map(file -> joinpath(dir, file), readdir(dir))

export readall
readall(f::Union{IO, AbstractString}, T) = reinterpret(T, read(f))

Serialization.serialize(file::AbstractString, x) = open(file, "w") do fid serialize(fid, x) end
Serialization.deserialize(file::AbstractString) = open(file, "r") do fid deserialize(fid) end

using Base: EachLine
export cycleline
function cycleline(filename::AbstractString; chomp=nothing, keep::Bool=false)
    if chomp !== nothing
        keep = !chomp
        depwarn("The `chomp=$chomp` argument to `eachline` is deprecated in favor of `keep=$keep`.", :eachline)
    end
    s = open(filename)
    EachLine(s, ondone=()->seekstart(s), keep=keep)::EachLine
end