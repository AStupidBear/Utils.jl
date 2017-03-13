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

using Utils, Knet, Hyperopt, JLD
export Knet, JLD

abstract Network
typealias KnetData Array{NTuple{2, Array{Float32}}}

export ClassifierParameter
@param immutable ClassifierParameter <: Parameter
  lr::Float64 = 0.01 => 0.01:0.01:0.5 # learning rate
  nlayers::Int = 3 => 2:4 # number of layers
  @rep 4 hidden::Int = 100 => 50:50:1000 # number of neurons
end

export Classifier
@with_kw type Classifier{T} <: Network
  nepochs::Int = 20 # number of epochs for traning
  maxevals::Int = 10 # max evaluations of hyperopt
  param::ClassifierParameter = ClassifierParameter()
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

function predict(w, x)
  for i in 1:2:length(w)
    x = w[i] * x .+ w[i+1]
    if i < length(w)-1
      x = relu(x) # max(0,x)
    end
  end
  return x
end

function loss(w, x, ygold)
  ypred = predict(w, eltype(w)(x))
  ynorm = logp(ypred, 1) # ypred .- log(sum(exp(ypred),1))
  -sum(ygold .* ynorm) / size(ygold, 2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr = 0.5, nepochs = 10)
  for epoch in 1:nepochs
    for (x, y) in dtrn
      g = lossgradient(w, x, y)
      for i in 1:length(w)
        axpy!(-lr, g[i], w[i])
      end
    end
  end
  return w
end

function train_policy(w, dtrn; lr = 0.5, nepochs = 10)
  for epoch in 1:nepochs
    for (x, r) in dtrn
      ypred = predict(w, eltype(w)(x))
      ypred .== maximum(ypred, 1)
      g = lossgradient(w, x, ypred)
      for i in 1:length(w)
        axpy!(-lr * mean(r), g[i], w[i])
      end
    end
  end
  return w
end

function accuracy(w, dtst, pred = predict)
  ncorrect = ninstance = nloss = 0
  for (x, ygold) in dtst
    ypred = pred(w, eltype(w)(x))
    ynorm = ypred .- log(sum(exp(ypred), 1))
    nloss += -sum(ygold .* ynorm)
    ncorrect += sum(ygold .* (ypred .== maximum(ypred, 1)))
    ninstance += size(ygold, 2)
  end
  return (ncorrect / ninstance, nloss / ninstance)
end

function weights(data, h...; atype = Array{Float32}, winit = 0.1)
  x0, y0 = data[1]
  w = Any[]
  x = size(x0, 1)
  for y in [h..., size(y0, 1)]
    push!(w, atype(winit * randn(y, x)))
    push!(w, atype(zeros(y, 1)))
    x = y
  end
  return w
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

function objective{T}(net::Classifier{T}, param)
    param = catparam(net.param, param)
    h = unroll(param, "hidden")[1:param.nlayers]
    w = weights(net.dtrn, h...; atype = T)

    train(w, net.dtrn, ; lr = param.lr, nepochs = net.nepochs)
    acc, loss = accuracy(w, net.dtst)

    file = tempname() * ".jld"; save(file, "w", w)
    T == Array{Float32} ? Knet.gc() : Knet.knetgc()
    display(param); @show acc

    return Dict("loss" => loss, "status" => STATUS_OK, "file" => file)
end

function train{T}(net::Classifier{T}, x, y; batchsize = 50)
  x = fit_transform(net.preprocs, x)
  y = fit_transform(net.encoder, y)
  xtrn, ytrn, xtst, ytst = splitdata(x, y)
  net.dtrn = minibatch(xtrn, ytrn, batchsize)
  net.dtst = minibatch(xtst, ytst, batchsize)

  fn = partial(objective, net)
  param, trials = optimize(fn, net.param.bounds...;
                    maxevals = net.maxevals)
  net.param = catparam(net.param, param)
  net.w = load(best_result(trials)["file"], "w")
end

function train_policy{T}(net::Classifier{T}, x, r; batchsize = 1)
  x = transform(net.preprocs, x)
  r = reshape(r, 1, length(r))
  dtrn = minibatch(x, r, batchsize)
  train_policy(net.w, dtrn; lr = net.param.lr, nepochs = net.nepochs)
end

function predict{T}(net::Classifier{T}, x)
  x = transform(net.preprocs, x)
  y = Array(predict(net.w, T(x)))
  probs = softmax(y)
  preds = inverse_transform(net.encoder, probs)
  return preds, probs
end

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
  import JLD: writeas, readas
  type KnetJLD; a::Array; end
  writeas(c::KnetArray) = KnetJLD(Array(c))
  readas(d::KnetJLD) = KnetArray(d.a)
end

end # end of module MLP
