export readabsdir
readabsdir(dir) = map(file -> joinpath(dir, file), readdir(dir))

export readall
readall(f::Union{IO, AbstractString}, T) = reinterpret(T, read(f))

export @npwrite
macro npwrite(f, x, eltype = :Float32)
    :(write($f, $(string(x)), convert(Array{$eltype}, permutedims($x)))) |> esc
end