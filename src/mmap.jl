export mzeros, mones

for (fname, felt) in ((:mzeros,:zero), (:mones,:one))
    @eval begin
        function $fname(T::Type, dims::Tuple)
            isdir(".mempool") || mkdir(".mempool")
            file = joinpath(".mempool", randstring())
            x = Mmap.mmap(file, Array{T, length(dims)}, dims)
            fill!(x, $felt(T))
        end
        $fname(dims::Tuple) = ($fname)(Float64, dims)
        $fname(T::Type, dims...) = $fname(T, dims)
        $fname(dims...) = $fname(dims)
    end
end
