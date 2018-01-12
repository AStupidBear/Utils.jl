export colvec, rowvec
colvec(x) = reshape(x, length(x), 1)
rowvec(x) = reshape(x, 1, length(x))

Base.delete!(a::Array, val) = deleteat!(a, findfirst(a, val))

export deleteall!, remove!, removeall!
deleteall!(a::Array, val)  = deleteat!(a, find(x -> x==val, a))
remove!(a::Array, val) = delete!(a, val)
removeall!(a::Array, val) = deleteall!(a, val)

Base.vec(x::Number) = fill(x, 1)
Base.vec(x::Symbol) = fill(x, 1)
Base.vec(x::Expr) = x.args

# """
# ```julia
# X = [([1,2,3],[4,5,6]), ([1,2,3], [4,5,6])]
# vcat(X...)
# ```
# """
for f in (:vcat, :hcat)
    @eval begin
        function Base.$f(X::Tuple...)
            ntuple(length(X[1])) do j
                $f([X[i][j] for i in 1:length(X)]...)
                # mapreduce(i -> X[i][j], $f, 1:length(X))
            end
        end
    end
end

export splat
splat(list) = [item for sublist in list for item in sublist]

export csize, clength, ccount, cview, cget, cset!, size2
csize(a) = (ndims(a) == 1 ? size(a) : size(a)[1:end-1])
csize(a, n) = tuple(csize(a)..., n) # size if you had n columns
clength(a) = (ndims(a) == 1 ? 1 : stride(a, ndims(a)))
ccount(a) = (ndims(a) == 1 ? length(a) : size(a, ndims(a)))
cview(a, i) = (ndims(a) == 1 ? a[i] : view(a, ntuple(i->(:), ndims(a) - 1)..., i))
cget(a, i) = (ndims(a) == 1 ? a[i] : getindex(a, ntuple(i->(:), ndims(a)-1)..., i))
cset!(a, x, i) = (ndims(a) == 1 ? (a[i] = x) : setindex!(a, x, ntuple(i->(:), ndims(a) - 1)..., i))
size2(y) = (nd = ndims(y); (nd == 1 ? (length(y), 1) : (stride(y, nd), size(y, nd)))) # size as a matrix
size2(y, i) = size2(y)[i]

export rsize, rlength, rcount, rview, rget, rset!, size1
rsize(a) = (ndims(a)==1 ? size(a) : size(a)[2:end])
rsize(a, n) = tuple(n, rsize(a)...) # size if you had n columns
rlength(a) = (ndims(a) == 1 ? length(a) : stride(a, ndims(a)))
rcount(a) = (ndims(a) == 1 ? length(a) : size(a, 1))
rview(a, i) = (ndims(a) == 1 ? a[i] : view(a, i, ntuple(i->(:), ndims(a) - 1)...))
rget(a, i) = (ndims(a) == 1 ? a[i] : getindex(a, i, ntuple(i->(:), ndims(a)-1)...))
rset!(a, x, i) = (ndims(a)==1 ? (a[i] = x) : setindex!(a, x, i, ntuple(i->(:), ndims(a)-1)...))
size1(y) = (nd = ndims(y); (nd == 1 ? (length(y), 1) : (size(y, 1), prod(size(y)[2:end])))) # size as a matrix
size1(y, i) = size1(y)[i]

for s in (:cget, :rget, :cview, :rview)
    @eval $s(as::Tuple, i) = tuple([$s(a, i) for a in as]...)
end
for s in (:cset!, :rset!)
    @eval $s(as::Tuple, xs::Tuple, i) = for (a, x) in zip(as, xs); $s(a, x, i); end
end

Base.transpose(x::Tuple) = transpose.(x)
Base.ctranspose(x::Tuple) = ctranspose.(x)

ccount(x::Tuple) = ccount(x[1])

# function sp_A_mul_B!(y, rowptr, colptr, I, J, A, x)
#     fill!(y, zero(eltype(y)))
#     for col in 1:length(colptr)-1
#         xc = x[col]
#         @inbounds for s = colptr[col] : (colptr[col+1]-1)
#             y[I[s]] += A[s] * xc
#         end
#     end
# end
