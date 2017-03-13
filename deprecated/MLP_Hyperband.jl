"""
    using Utils, MLDatasets
    reload("MLP")

    x, y = MNIST.traindata()
    x = reshape(x, 28*28, 60000)
    x = x[1:10, 1:200]; y = y[1:200]

    net = MLP.Classifier()
    MLP.train(net, x, y)
    MLP.train_policy(net, x, y)
    preds, probs = MLP.predict(net, x)
"""
module MLP

using Utils, Knet, JLD
export Knet, JLD

abstract Network
typealias KnetData Array{NTuple{2, Array{Float32}}}

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
  import JLD: writeas, readas
  type KnetJLD; a::Array; end
  writeas(c::KnetArray) = KnetJLD(Array(c))
  readas(d::KnetJLD) = KnetArray(d.a)
end

###############################################################################
# general functions
###############################################################################
function mlp(w, x; pdrop = (0, 0))
  x = dropout(x, pdrop[1])
  for i = 1:2:(length(w) - 2)
      x = relu(w[i] * x .+ w[i+1])
      x = dropout(x, pdrop[2])
  end
  return w[end-1] * x .+ w[end]
end

function dropout(x,p)
    if p == 0
        x
    else
        x .* (rand!(similar(AutoGrad.getval(x))) .> p)
    end
end

function softloss(w, x, ygold, model; l1 = 0, l2 = 0, o...)
  ypred = model(w, eltype(w)(x))
  ynorm = logp(ypred, 1) # ypred .- log(sum(exp(ypred),1))
  J = -sum(ygold .* ynorm) / size(ygold, 2)
  if l1 != 0; J += l1 * sum(sumabs(wi)  for wi in w[1:2:end]); end
  if l2 != 0; J += l2 * sum(sumabs2(wi) for wi in w[1:2:end]); end
  return J
end
softgrad = grad(softloss)

function avgloss(w, data, model, lossfun)
    sum = cnt = 0
    for (x,y) in data
        sum += lossfun(w, x, y, model)
        cnt += 1
    end
    return sum / cnt
end

function zeroone(w, data, model)
    ncorr = ninst = 0
    for (x, y) in data
        ypred = model(w, x)
        ncorr += sum(y .* (ypred .== maximum(ypred, 1)))
        ninst += size(x, 2)
    end
    return 1 - ncorr / ninst
end

function weights(data, h...; atype = Array{Float32}, std = 0.01, seed = 1)
  r = MersenneTwister(seed)
  x, y = data[1]
  h = [size(x, 1), h..., size(y, 1)]
  w = Any[]
  for i=1:(length(h) - 1)
    push!(w, std * randn(r, h[i+1], h[i]))
    push!(w, zeros(h[i+1], 1))
  end
  map(atype, w)
end

function minibatch(x, y, batchsize = 50; atype = Array{Float32})
  x = atype(x); y = atype(y)
  data = Any[]
  for i in 1:batchsize:size(x, 2)
    j = min(i + batchsize - 1, size(x, 2))
    push!(data, (x[:, i:j], y[:, i:j]))
  end
  return data
end

function train(w, p, dtrn, dtst, model, lossfun; epochs = 100, o...)
  gradfun = grad(lossfun)
  best = (0, deepcopy(w), avgloss(w, dtst, model, lossfun))
  for epoch in 1:epochs
    for (x, y) in dtrn
      g = gradfun(w, x, y, model; o...)
      update!(w, g, p)
    end
    ltst = avgloss(w, dtst, model, lossfun)
    if ltst < best[3]
      best = (epoch, deepcopy(w), ltst)
    end
  end
  (epoch, wbest, ltst) = best
  ltrn = avgloss(wbest, dtrn, model, lossfun)
  return (epoch, ltrn, ltst)
end

function getconfig()
    pmax = 0.5
    hmin, hmax = 32, 512
    pdrop = (rand() < 1/3 ? (rand() * pmax, rand() * pmax) :
             rand() < 1/2 ? (rand() * pmax, 0) : (0, rand() * pmax))
    h() = round(Int, hmin + (hmax - hmin)^rand())
    nlayer = rand(2:4)
    hidden = ntuple(x->h(), nlayer)
    return (hidden, pdrop)
end

# function train_policy(w, dtrn; lr = 0.5, nepochs = 10)
#   for epoch in 1:nepochs
#     for (x, r) in dtrn
#       ypred = predict(w, eltype(w)(x))
#       ypred .== maximum(ypred, 1)
#       g = softgrad(w, x, ypred)
#       for i in 1:length(w)
#         axpy!(-lr * mean(r), g[i], w[i])
#       end
#     end
#   end
#   return w
# end

###############################################################################
# end of general functions
###############################################################################

###############################################################################
# Classifier
###############################################################################
export Classifier
@with_kw type Classifier{T} <: Network
  nepochs::Int = 20 # number of epochs for traning
  maxevals::Int = 10 # max evaluations of hyperopt
  encoder::OneHotEncoder = OneHotEncoder()
  preprocs::MinMaxScaler = MinMaxScaler()
  w::Array{T} = []
  dtrn::KnetData = []
  dtst::KnetData = []
end

function Classifier(; gpu = false, kwargs...)
  atype = gpu ? KnetArray{Float32} : Array{Float32}
  Classifier{atype}(; kwargs...)
end

function getloss{T}(net::Classifier{T}, c, n)
  w = weights(net.dtrn, c[1]...; atype = T)
  p = [Adam() for wi in w]
  (epoch, ltrn, ltst) = train(w, p, net.dtrn, net.dtst, mlp, softloss;
                              epochs = n, pdrop = c[2])
  report(net, w, c, epoch)
  return ltst
end

function report(net::Classifier, w, c, epoch)
  acc_trn = 1 - zeroone(w, net.dtrn, mlp)
  acc_tst = 1 - zeroone(w, net.dtst, mlp)
  @DataFrame(epoch, acc_trn, acc_tst, c) |> println
end

function retrain!{T}(net::Classifier{T}, best)
  ~, n, c = best
  w = weights(net.dtrn, c[1]...; atype = T)
  p = [Adam() for wi in w]
  (epoch, ltrn, ltst) = train(w, p, net.dtrn, net.dtst, mlp, softloss;
                              epochs = n, pdrop = c[2])
  net.w = deepcopy(w)
end

function train!{T}(net::Classifier{T}, x, y; batchsize = 50)
  x = fit_transform(net.preprocs, x)
  y = fit_transform(net.encoder, y)
  xtrn, ytrn, xtst, ytst = splitdata(x, y)
  net.dtrn = minibatch(xtrn, ytrn, batchsize)
  net.dtst = minibatch(xtst, ytst, batchsize)
  best = hyperband(getconfig, partial(getloss, net))
  retrain!(net, best)
end

# function train_policy{T}(net::Classifier{T}, x, r; batchsize = 1)
#   x = transform(net.preprocs, x)
#   r = reshape(r, 1, length(r))
#   dtrn = minibatch(x, r, batchsize)
#   train_policy(net.w, dtrn; lr = net.param.lr, nepochs = net.nepochs)
# end

function predict(net::Classifier, x)
  x = transform(net.preprocs, x)
  y = Array(mlp(net.w, eltype(net.w)(x)))
  probs = softmax(y)
  preds = inverse_transform(net.encoder, probs)
  return preds, probs
end
###############################################################################
# end of Classifier
###############################################################################
end # end of module MLP
