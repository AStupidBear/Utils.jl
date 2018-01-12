export readabsdir
readabsdir(dir) = map(file -> joinpath(dir, file), readdir(dir))

export readall
readall(f::Union{IO, AbstractString}, T) = reinterpret(T, read(f))
