"""
    import Boost, MNIST
    xtrn, ytrn = MNIST.traindata()
    xtst, ytst = MNIST.testdata()
    x = [xtrn xtst]
    y = [ytrn;ytst]
    x = x[1:10,1:20]
    y = y[1:20]
    bst, args, space, loss = Boost.main(x,y)
    preds = Boost.predict(bst, x)
"""
module Boost

using Utils, XGBoost, HyperOpt

abstract Booster

function train!(bst::Booster, x, y; method = 0)
  preprocess!(bst, x, y)
  hyperoptimize!(bst, Val{method})
end

function preprocess!(clf::Classifier, x, y)
  x, y = shuffle(x, y)
  x, y = balance(x, y)
  y = fit_transform(bst.encoder, y)
  clf.x, clf.y = x, y
end

preprocess!(reg::Regressor, x, y) = (clf.x, clf.y = shuffle(x, y))

function hyperoptimize!(bst::Booster, ::Type{Val{0}})
  c = getparam(bst.param)
  bst.model = 0
end

function hyperoptimize!(bst::Booster, ::Type{Val{:Hyperband}})
  budget = bst.maxevals * bst.epochs
  best = Hyperband.hyperband(partial(getconfig, bst),
                            partial(getloss, bst), budget)
  ~, ~, cmin = best
  bst.param = catparam(bst.param, splat(cmin))
  bst.model = 0
end

function hyperoptimize!(bst::Booster, ::Type{Val{:BO}})
  bounds = bst.param.bounds
  c0 = getparam(bst.param)
  fn = partial(getloss, bst)
  cmin, ymin, progress = BO.minimize(fn, bounds, c0;
                          name = "Booster_Classifier",
                          maxevals = bst.maxevals)
  bst.param = catparam(bst.param, cmin)
  bst.model = 0
end

function hyperoptimize!(bst::Booster, ::Type{Val{:TPE}})
  bounds = bst.param.bounds
  cmin, trials = Hyperopt.minimize(partial(getloss, bst),
                        bounds; maxevals = bst.maxevals)
  bst.param = catparam(bst.param, cmin)
  bst.model = 0
end


@param immutable GBParameter
  max_depth::Int = 5 => 1:20
  eta::Float64 = 1 => (0.5, 1)
end

@with_kw type Classifier
  param::GBParameter = GBParameter()

  maxevals::Int = 1 # max evaluations of hyperopt
  epochs::Int = 1

  encoder::LabelEncoder = LabelEncoder()

  model = nothing
  x::Array{Float64, 2} = zeros(0, 0)
  y::Array{Float64, 1} = zeros(0)
end

function fit(bst::Booster, c, n)
  max_depth, eta = c

  param = Dict("max_depth" => max_depth, "eta" => eta,
              "num_class"=> length(bst.encoder.unique_label),
              "silent" => 2, "objective" => "multi:softmax")
  bst = nfold_cv(x, epochs, label = y, param = param, metrics = ["auc"])
  preds = XGBoost.predict(bst, o[:xtst])
  loss = mean(preds .!= o[:ytst])

  @printf("\nepochs=%d,max_depth=%d,eta=%6.4f,loss=%6.4f\n",
  epochs,max_depth,eta,loss)

  report(bst, w, epoch)
  return model, ltst
end

function fit(bst::Booster, c)
  param = catparam(bst.param, c)
  nhid = param.nhid
  hidden = unroll(param, "hidden")[1:nhid]
  pdrop = unroll(param, "pdrop")

  w = weights(bst.model, bst.dtrn, bst.atype, hidden)
  p = [bst.updater() for wi in w]
  (epoch, ltrn, ltst) = train(w, p, bst.dtrn, bst.dtst, bst.model,
                        bst.loss; epochs = bst.epochs, pdrop = pdrop)
  report(bst, w, epoch)
  return w, ltst
end

function report(clf::Classifier, epoch)
  @DataFrame(epoch, max_depth, eta, loss) |> debug
end

function getconfig(bst::Booster)
    nhid = rand(1:6)

    hmin, hmax = 32, 1024
    h() = round(Int, hmin + (hmax - hmin)^rand())
    hidden = ntuple(x->h(), 6)

    pmax = 0.5
    pdrop = (rand() < 1/3 ? (rand() * pmax, rand() * pmax) :
    rand() < 1/2 ? (rand() * pmax, 0) : (0, rand() * pmax))

    return nhid, hidden, pdrop
end

function objective(args)

    param = Dict("max_depth"=>max_depth,"eta"=>eta, "num_class"=>o[:nclass],
    "silent"=>2, "objective"=>"multi:softmax")
    bst = xgboost(o[:xtrn], epochs, label=o[:ytrn],param=param)
    preds = XGBoost.predict(bst, o[:xtst])
    loss = mean(preds .!= o[:ytst])

    @printf("\nepochs=%d,max_depth=%d,eta=%6.4f,loss=%6.4f\n",
    epochs,max_depth,eta,loss)

end


function predict(clf::Classifier, x)
  x = transform(bst.preprocs, x)
  ypred = inverse_transform(clf.encoder, XGBoost.predict(clf.bst, x))
end

predict(reg::Regressor, x) = XGBoost.predict(reg.bst, x)

function test(clf::Classifier, x, y)
  ypred = predict(clf, x)
  sum(y .== ypred) / length(y)
end

function test(reg::Regressor, x, y)
  ypred = predict(reg, x)
  sumabs(y - ypred) / length(y)
end

train_X = rand(100, 10)
train_Y = rand(0:1, 100)
nfold = 5
param = ["max_depth" => 2,
         "eta" => 1,
         "objective" => "binary:logistic"]
metrics = ["auc"]
nfold_cv(train_X, 2, nfold, label = train_Y, param = param, metrics = metrics)

end # end of module Boost
