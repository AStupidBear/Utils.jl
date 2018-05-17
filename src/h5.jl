using HDF5

Base.basename(file::HDF5.HDF5Dataset) = basename(HDF5.name(file))

export h5cat
function h5cat(src, dst)
    fsrc, fdst = h5open(src, "r"), h5open(dst, "w")
    for dset in first(fsrc)
        dims = dataspace(size(dset)..., Int(length(fsrc)))
        d_create(fdst, basename(dset), datatype(eltype(dset)), dims)
    end
    for (i, g) in enumerate(fsrc)
        for dset in g
            cset!(fdst[basename(dset)], read(dset), i)
        end
    end
    close(fsrc)
    close(fdst)
end

export @npwrite
macro npwrite(f, x, eltype = :Float32)
    :(write($f, $(string(x)), convert(Array{$eltype}, permutedims($x)))) |> esc
end