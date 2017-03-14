"""
    N = 10; t = linspace(0,2π,N); x = cos(t); y = sin(t)
    data = hcat([x y]',[x+4 y]')
    reload("TDA")
    TDA.perseus(data; a_max=2.0, dim_max=2, plot=true)
"""

module TDA

using Gadfly, LightGraphs, Combinatorics
import Base.+
+(x::UInt8,y::UInt8) = x $ y
isocc(v) = !isempty(v.rowval)
top(v) = v.rowval[end]

function col_reduce(∂) # column reduction from boundary matrix
    _top = Dict{Int,Int}()
    R = SparseMatrixCSC{UInt8,Int64}[∂[:,i] for i=1:size(∂,2)]

    for j=1:length(R)
        while isocc(R[j])
            i=top(R[j])
            if haskey(_top,i)
                R[j] += R[_top[i]]
            else
                _top[i] = j
                break
            end
        end
    end
    R
end

function threshold!(D, a, σ)
    n = length(σ)
    if n == 1
        a[σ] = 1e-5
    elseif n ==2
        a[σ] = D[σ...]
    else
        r = 0.0
        for δ in combinations(σ,n-1)
            r = max(r,a[δ])
        end
        a[σ] = r + 1e-5
    end
end

function perseus(data; a_max=1.0, dim_max=2, plot=false)
  M = size(data, 2)
  D = [norm(data[:,i] - data[:,j]) for i=1:M, j=1:M]

  g = Graph(Int.(D .< a_max)) # gplot(g) |> display
  cliques = maximal_cliques(g)
  dim_max = min(dim_max, maximum(map(length,cliques))-1)

  filtrations = Array{Array{Int64,1},1}()
  for dim in 0:dim_max
      for clique in cliques
            append!(filtrations,collect(combinations(clique,dim+1)))
      end
  end

  a = Dict{Vector{Int},Float64}() # threshold radius of σs
  for σ in filtrations
      threshold!(D, a, σ)
  end

  p = sortperm(map(x->a[x],filtrations))
  filtrations = filtrations[p]
  index = Dict{Vector{Int},Int}()
  for (i,σ) in enumerate(filtrations)
      index[σ] = i
  end

  ∂ = spzeros(UInt8,length(filtrations),length(filtrations))
  for (j,σ) in enumerate(filtrations)
      n = length(σ)
      if n ==1; continue; end
      for δ in combinations(σ,n-1)
          ∂[index[δ],j] = one(UInt8)
      end
  end

  R = col_reduce(∂)

  birth = Float64[]
  death = Float64[]
  degree = Int[]
  for j = 1:length(R)
      if !isempty(R[j].rowval)
          i = R[j].rowval[end] # birth time
          push!(birth,a[filtrations[i]])
          push!(death,a[filtrations[j]])
          push!(degree, length(filtrations[i])-1) # degree
      end
  end
  b0 = count(x->length(x)==1,filtrations)-sum(degree.==0)
  append!(birth,zeros(b0))
  append!(death,a_max*ones(b0))
  append!(degree, zeros(Int,b0))
  plot==true && persdia(birth, death, degree, a_max)
end

function persdia(birth, death, degree, a_max)
  p = plot(layer(x = birth, y = death, color = string.(degree),
  Geom.beeswarm(orientation=:horizontal)),
  layer(x = [0,a_max], ymin = [0,0], ymax = [0,a_max], Geom.ribbon),
  Guide.xlabel("birth"), Guide.ylabel("death"), Guide.colorkey("Betti Number"),
  Theme(default_point_size=1.5pt))
end

end # end of TDA
