export histn
"c, w = histn(rand(10), rand(10), rand(10); nbins = 10)"
function histn(xs::Array...; o...)
    h = StatsBase.fit(StatsBase.Histogram, vec.(xs); closed = :left, o...)
    edge2center.(h.edges), h.weights
end

export edge2center
"edge2center(0:0.1:1)"
function edge2center(edge)
    dx = edge[2] - edge[1]
    center = edge[1:(end - 1)] + dx / 2
end

export xcov
xcov(A,B) = xcorr(A .- mean(A), B .- mean(B))

# export entropy
# "entropy(randn(10000))"
# function entropy(x)
#     ε = 1e-100
#     c, w = histn(x)
#     P = w / sum(w)
#     H = sum(-P .* log.(2, P .+ ε))
#     scale = log(2, prod(length.(c)))
#     return H / scale
# end
#
# export mutualinfo
# "mutualinfo(randn(10000), randn(10000))"
# function mutualinfo(x, y)
#     ε = 1e-100
#     c, w = histn(x, y)
#     Pxy = w / sum(w)
#     Px = sum(Pxy, 2)
#     Py = sum(Pxy, 1)
#     Hx = sum(-Px .* log.(2, Px .+ ε))
#     Hy = sum(-Py .* log.(2, Py .+ ε))
#     I = sum(Pxy .* log.(2, Pxy ./ (Px .* Py .+ ε) .+ ε))
#     return I / Hy
# end
#
# export kl
# "kl(randn(100000), randn(100000) + 10)"
# function kl(x1, x2)
#     ε = 1e-100
#     low, up = extrema((extrema(x1)..., extrema(x2)...))
#     edge = linspace(low, up, 1000)
#     P1 = normalize(fit(Histogram, vec(x1), edge), 1)
#     P2 = normalize(fit(Histogram, vec(x2), edge), 1)
#     KL = sum(P1 .* log.(2, P1 ./ (P2 .+ ε) .+ ε))
# end

# export randprob
# "randprob([-1, 1], [0.2, 0.8], 10, 10)"
# function randprob(list, prob)
#     Z = 1.0 # partition number
#     for i in eachindex(list)
#         rand() < prob[i] / Z && return list[i]
#         Z -= prob[i]
#     end
# end
#
# function randprob(list, prob, dims...)
#     x = zeros(dims...)
#     for i in eachindex(x)
#         x[i] = randprob(list, prob)
#     end
#     x
# end

# """
# ```
# using Distributions
# rand(MvNormal([1.,2.,3.], [1., 1., 1.]), 1000)
# ```
# """
# function pca(data::Array{Float64,2}, n=1)
#     N = size(data, 2)
#     data = data .- mean(data, 2)
#     data_cov =1 / N * data * data.'
#     D, V = eig(data_cov)
#     idx = sortperm(D, rev=true)
#     D = D[idx]
#     V = V[:,idx]
#     data_new = V.'[1:n,:] * data
#     return D[1:n], V[:,1:n]
# end
#
# """
# ```
# addprocs(3)
# T = 1000000
# s1 = 2 * rand(T) - 1;	s2 = 2 * rand(T) - 1
# S0 = [s1'; s2']
# X = S0 * [cos(π/4) sin(π/4); -sin(π/4) cos(π/4)]
# @time A = ica(X, 2)
# ```
# """
# function ica(X::Array{Float64,2}, M::Int64)
#     X = X.- mean(X,2)
#     N, T = size(X)
#     Σ = 1 / T * X * X'
#     d, U = eig(Σ)
#     d = d[1:M]; U = U[:,1:M]
#     Q = diagm(d.^(-1/2)) * U'
#     Z = Q * X
#     W = zeros(M, N)
#
#     for k = 1:M
#         w = randn(N); w = w / norm(w)
#         Δ = 1.0
#         while Δ > 1e-3
#             w0 = w
#             expectation = 1 / T * @parallel (+) for i=1:1:T
#             dot(w,Z[:,i])^3*Z[:,i]
#         end
#         w = expectation - 3w
#         if k >= 2
#             w = w - W[1:k-1,:]' * W[1:k-1,:] * w
#         end
#         w = w / norm(w)
#         Δ = 1 - abs(dot(w, w0))
#     end
#     W[k, :] = w'
# end
# return W * Q
# end
