export readabsdir
readabsdir(dir) = map(file -> joinpath(dir, file), readdir(dir))

export readall
readall(f::Union{IO, AbstractString}, T) = reinterpret(T, read(f))

Serialization.serialize(file::AbstractString, x) = open(file, "w") do fid serialize(fid, x) end
Serialization.deserialize(file::AbstractString) = open(file, "r") do fid deserialize(fid) end