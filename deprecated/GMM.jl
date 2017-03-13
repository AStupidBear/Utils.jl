"""
    x = [randn(100); 5 + randn(100)]
    @show μ, σ2, label = GMM2.gmm(x; K=2)
"""
module GMM2

pdf(x, μ, σ2) = exp(-0.5 * (x - μ) ^ 2 / σ2)

function initialize(x, K)
  N = length(x)
  μ = K == 1 ? fill(mean(x),K) : quantile(x, linspace(0.2,0.8,K))
  σ2 = fill(var(x), K)
  p = ones(K)/K
  w = ones(N, K)/K
  w, μ, σ2, p
end

function Estep!(x, w, μ, σ2, p)
  N = length(x)
  K = length(μ)
  for k = 1:K
    for i in 1:N
      w[i,k] = pdf(x[i],μ[k],σ2[k]) * p[k]
    end
  end
  w ./= sum(w, 2)
end

function Mstep!(x, w, μ, σ2, p)
  N = length(x)
  K = length(μ)
  for k = 1:K
      p[k] = sum(w[i,k] for i in 1:N)
      μ[k] = sum(w[i,k] * x[i] for i in 1:N) / p[k]
      σ2[k] = sum(w[i,k] *(x[i] - μ[k]) ^ 2 for i in 1:N) / p[k]
  end
  normalize!(p, 1)
end


function gmm(x; K=1, MaxIter=1000)
  w, μ, σ2, p = initialize(x, K)
  for iter = 1:MaxIter
    Estep!(x, w, μ, σ2, p)
    Mstep!(x, w, μ, σ2, p)
  end
  label = [indmax(w[i,:]) for i =1:length(x)]
  μ, sqrt(σ2), p, label
end

end
