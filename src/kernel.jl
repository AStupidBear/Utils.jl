export Nystroem, fit, transform
type Nystroem{T}
    kernel::Kernel{T}
    n_components::Int
    random_state::Int
    normalization::Matrix{T}
    components::Matrix{T}
end

Nystroem(;γ = 0.2, n_components = 30, random_state = 0) = Nystroem(GaussianKernel(γ), n_components, random_state, zeros(typeof(γ), 0, 0), zeros(typeof(γ), 0, 0))

@noinline function transform(estimator::Nystroem, X)
    embedded = kernelmatrix(estimator.kernel, X, estimator.components)
    A_mul_Bt(embedded, estimator.normalization)
end

function fit{T}(estimator::Nystroem{T}, X)
    X = unique(X, 1)
    srand(estimator.random_state)
    n_samples = size(X, 1)
    n_components = min(n_samples, estimator.n_components)
    basis_inds = randperm(n_samples)[1:n_components]
    basis = X[basis_inds, :]
    K = kernelmatrix(estimator.kernel, basis)
    U, S, V = svd(K); S = max(S, 1e-12)
    estimator.normalization = A_mul_Bt(U ./ sqrt(S.'), V)
    estimator.components = basis
end
