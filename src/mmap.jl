export mzeros, mones, mcopy

for (fname, felt) in ((:mzeros,:zero), (:mones,:one))
    @eval begin
        $fname(a::AbstractArray, T::Type, dims::Tuple) = $fname(T, dims)
        $fname(a::AbstractArray, T::Type, dims...) = $fname(T, dims...)
        $fname(a::AbstractArray, T::Type = eltype(a)) = $fname(T, size(a))
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

mcopy(x) = copy!(mzeros(x), x)
