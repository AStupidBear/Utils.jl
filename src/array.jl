export colvec, rowvec
colvec(x) = reshape(x, length(x), 1)
rowvec(x) = reshape(x, 1, length(x))

Base.delete!(a::Array, val) = deleteat!(a, findfirst(a, val))

export deleteall!, remove!, removeall!
deleteall!(a::Array, val)  = deleteat!(a, find(x -> x == val, a))
remove!(a::Array, val) = delete!(a, val)
removeall!(a::Array, val) = deleteall!(a, val)

Base.vec(x::Number) = fill(x, 1)
Base.vec(x::Symbol) = fill(x, 1)
Base.vec(x::Expr) = x.args

Base.vcat(xs::Tuple...) = map(vcat, xs...)
Base.hcat(xs::Tuple...) = map(hcat, xs...)

export splat
splat(list) = [item for sublist in list for item in (isa(sublist, AbstractArray) ? sublist : [sublist])]

export csize, rsize
csize(a) = (ndims(a) == 1 ? size(a) : size(a)[1:end-1])
csize(a, n) = tuple(csize(a)..., n) # size if you had n columns
rsize(a) = (ndims(a) == 1 ? size(a) : size(a)[2:end])
rsize(a, n) = tuple(n, rsize(a)...) # size if you had n columns

export clength, rlength
clength(a) = (ndims(a) == 1 ? 1 : stride(a, ndims(a)))
rlength(a) = (ndims(a) == 1 ? length(a) : stride(a, ndims(a)))

export ccount, rcount
ccount(a) = (ndims(a) == 1 ? length(a) : size(a, ndims(a)))
rcount(a) = (ndims(a) == 1 ? length(a) : size(a, 1))

export size2, size1
size2(y) = (nd = ndims(y); (nd == 1 ? (length(y), 1) : (stride(y, nd), size(y, nd)))) # size as a matrix
size2(y, i) = size2(y)[i]
size1(y) = (nd = ndims(y); (nd == 1 ? (length(y), 1) : (size(y, 1), prod(size(y)[2:end])))) # size as a matrix
size1(y, i) = size1(y)[i]

@generated function subslice(x::AbstractArray{T, N}) where {T, N}
    inds = ntuple(i -> (:), N - 1)
    :($inds)
end
subslice(x) = ntuple(i -> (:), ndims(x) - 1)

export cview, rview, cget, rget, cset!, rset!
cview(a, i) = view(a, subslice(a)..., i)
rview(a, i) = view(a, i, subslice(a)...)
cget(a, i) = getindex(a, subslice(a)..., i)
rget(a, i) = getindex(a, i, subslice(a)...)
cset!(a, x, i) = setindex!(a, x, subslice(a)..., i)
rset!(a, x, i) = setindex!(a, x, i, subslice(a)...)

for s in (:cget, :rget, :cview, :rview)
    @eval $s(as::Tuple, i) = tuple([$s(a, i) for a in as]...)
end
for s in (:cset!, :rset!)
    @eval $s(as::Tuple, xs::Tuple, i) = for (a, x) in zip(as, xs); $s(a, x, i); end
end

Base.transpose(x::Tuple) = transpose.(x)

ccount(x::Tuple) = ccount(x[1])
rcount(x::Tuple) = rcount(x[1])

_view(x, idx1, idx2, D) = (view(x, idx1..., i, idx2...) for i in axes(x, D))
@generated function Base.eachslice(x::AbstractArray{T, N}, ::Type{Val{D}}) where {T, N, D}
    idx1, idx2 = ntuple(d -> (:), D - 1), ntuple(d -> (:), N - D)
    :(_view(x, $idx1, $idx2, $D))
end

export unsqueeze
unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim - 1]..., 1, size(xs)[dim:end]...))

Base.permutedims(x::AbstractArray) = permutedims(x, ndims(x):-1:1)
Base.PermutedDimsArray(x::AbstractArray) = PermutedDimsArray(x, ndims(x):-1:1)

function Base.cat(xs::Vector{<:AbstractArray{T, N}}, dim::Integer) where {T, N}
    ysize = ntuple(i -> i != dim ? size(first(xs), i) : sum(size(x, dim) for x in xs), Val(N))
    y = zeros(T, ysize)
    pos = 0
    for x in xs
        slice = (pos + 1):(pos + size(x, dim))
        inds = ntuple(@closure(i -> i != dim ? (1:size(y, i)) : slice), Val(N))
        copyto!(view(y, inds...), x)
        pos += size(x, dim)
    end
    return y
end

export stack, unstack
stack(xs::Vector{<:AbstractArray}, dim::Integer) = cat(unsqueeze.(xs, dim), dim)
unstack(xs, dim::Integer) = [copy(selectdim(xs, dim, i)) for i in 1:size(xs, dim)]
stack(xs::AbstractArray...; dims) = stack(collect(xs), dims)

export cstack, rstack
cstack(xs::Vector{<:AbstractArray}) = stack(xs, ndims(first(xs)) + 1)
rstack(xs::Vector{<:AbstractArray}) = stack(xs, 1)
cstack(xs::AbstractArray...) = cstack(collect(xs))
rstack(xs::AbstractArray...) = rstack(collect(xs))

export zeroel, oneel
zeroel(x) = zero(eltype(x))
oneel(x) = one(eltype(x))

export countunique
countunique(itr) = length(Set(itr))

export rreshape, creshape
rreshape(x) = reshape(x, size(x, 1), :)
creshape(x) = reshape(x, :, size(x)[end])

Base.dropdims(f, A::AbstractArray; dims) = dropdims(f(A, dims = dims), dims = dims)

# function sp_A_mul_B!(y, rowptr, colptr, I, J, A, x)
#     fill!(y, zero(eltype(y)))
#     for col in 1:length(colptr)-1
#         xc = x[col]
#         @inbounds for s = colptr[col] : (colptr[col+1]-1)
#             y[I[s]] += A[s] * xc
#         end
#     end
# end