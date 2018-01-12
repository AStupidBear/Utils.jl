export ndgrid

function ndgrid{T}(vs::AbstractVector{T}...; mat=false, rev=false)
    n = length(vs)
    sz = map(length, vs)
    if mat == false
        out = ntuple(i->Array{T}(prod(sz)), n)
    else
        out = ntuple(i->Array{T}(sz), n)
    end
    s = 1
    for i=1:n
        if rev==true
            a = out[i]::Array
        else
            a = out[n-i+1]::Array
        end
        v = vs[i]
        snext = s*sz[i]
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end

function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end
