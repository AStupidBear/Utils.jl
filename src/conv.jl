"""
```julia
conv(ones(3)/3, rand(100), :origin)
```
"""
function Base.conv(u, v, o = :origin)
    Base.conv(u, v)[1:length(v)]
end

"""
```julia
A = ones(5, 5)
B = [-1. -1 -1;
    -1 +8  -1;
    -1 -1 -1] / 8
C = conv2(A, B, :origin)
```
"""
function Base.conv2(A::Array{Float64, 2}, B::Array{Float64, 2}, o = :origin)
    n1, n2 = size(A)
    m1, m2 = size(B)
    C = Base.conv2(A, B)
    return C[m1-1:m1+n1-2, m2-1:m2+n2-2]
end
