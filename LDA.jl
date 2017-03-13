"""
    X = [3 2 2 1 4 6 8 9 9 10;
         6 4 3 4 4 9 7 5 10 8]

    y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    @show LDA.fisher_score(X, y, [1.0, 0.0])
    @show LDA.fisher_score(X, y, [0.0, 1.0])

    D, V = LDA.lda(X, y)

    max_score = D[2]
    k = V[2, 2] / V[2, 1]

    import Plots
    Plots.scatter(X[1,:], X[2,:], group=y, leg=:none)
    xmin, xmax = extrema(X[1,:])
    ymin, ymax = extrema(X[2,:])
    x = linspace(xmin, xmax, 100)
    Plots.plot!(x, k * (x - xmin) + ymin)
"""

module LDA

function fisher_score(X, y, e)
  x = At_mul_B(normalize(e), X)

  N = length(x)
  Nc = maximum(y)

  n = zeros(Nc)
  μ = zeros(Nc)
  σ2 = zeros(Nc)

  for i in 1:N
     c =  y[i]
     n[c] += 1
     μ[c] += x[i]
  end
  μ ./= n

  for i in 1:N
    c = y[i]
    σ2[c] += (x[i] - μ[c]) ^ 2
  end
  σ2 ./= n

  μtot = mean(x)
  sb = sum(n[c] * (μ[c] - μtot) ^ 2 for c in 1:Nc)
  sw = sum(n[c] * σ2[c] for c in 1:Nc)
  score = sb / sw
end

function lda(X, y)
  D, N = size(X)
  Nc = maximum(y)

  n = zeros(Nc)
  μ = [zeros(D) for c in 1:Nc]
  Σ = [zeros(D, D) for c in 1:Nc]

  for i = 1:N
    c = y[i]
    n[c] += 1
    for d = 1:D
      μ[c][d] += X[d, i]
    end
  end
  μ ./= n

  for i in 1:N
    c = y[i]
    for d1 = 1:D, d2 = 1:D
      Σ[c][d1, d2] += (X[d1, i] - μ[c][d1]) * ((X[d2, i] - μ[c][d2]))
    end
  end
  Σ ./= n

  μtot = mean(X, 2)

  Sw = zeros(D, D)
  for c = 1:Nc, d2 = 1:D, d1 = 1:D
    Sw[d1, d2] += n[c] * Σ[c][d1, d2]
  end

  Sb = zeros(D, D)
  for c = 1:Nc, d2 = 1:D, d1 = 1:D
    Sb[d1, d2] += n[c] * (μ[c][d1] - μtot[d1]) * (μ[c][d2] - μtot[d2])
  end

  D, V = eig(Sb, Sw)
end

end
