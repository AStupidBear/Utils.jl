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

o = Dict()
o[:max_depth_min] = 5
o[:max_depth_step] = 2
o[:max_depth_max] = 20
o[:num_round_min] = 5
o[:num_round_step] = 2
o[:num_round_max] = 20
o[:η_min] = 0.5
o[:η_max] = 1.0
o[:maxevals]= 10

using XGBoost, Hyperopt
export XGBoost

function splitdata(x, y)
    n1, n2 = size(x)
    xtrn = x[1:4*n1÷5, :]
    xtst = x[4*n1÷5+1:n1, :]
    ytrn = y[1:4*n1÷5]
    ytst = y[4*n1÷5+1:n1]
    return  xtrn, ytrn, xtst, ytst
end

function label2int(l)
    if !haskey(o,:lu)
      o[:lu] = unique(l)
      eltype(o[:lu])<:Number && (o[:lu] = sort(o[:lu]))
    end
    lu = o[:lu]
    I = zeros(Int,l)
    for i in eachindex(I)
        I[i] = find(lu .== l[i])[1] - 1
    end
    I
end

function int2label(I)
    lu = o[:lu]
    I = Int.(I)
    l = zeros(I)
    for i in eachindex(I)
        l[i] = lu[I[i]+1]
    end
    l
end

function preprocess(x,y)
    y = vec(y)
    y = label2int(y)
    return x,y
end

function objective(args)
    num_round,max_depth,η = args
    num_round = Int(num_round)
    max_depth = Int(max_depth)

    param = Dict("max_depth"=>max_depth,"eta"=>η, "num_class"=>o[:nclass],
    "silent"=>2, "objective"=>"multi:softmax")
    bst = xgboost(o[:xtrn], num_round, label=o[:ytrn],param=param)
    preds = XGBoost.predict(bst, o[:xtst])
    loss = mean(preds .!= o[:ytst])

    @printf("\nnum_round=%d,max_depth=%d,η=%6.4f,loss=%6.4f\n",
    num_round,max_depth,η,loss)

    o[:bestfile] = tempname()*".model"
    save(bst, o[:bestfile])
    return Dict("loss" => loss, "status" => STATUS_OK,
    "bst_file" => o[:bestfile], "space" => args)
end

export main
function main(x, y; batchsize=50, gpu=false)
    x = x'
    x, y = preprocess(x, y)
    o[:xtrn], o[:ytrn], o[:xtst], o[:ytst] = splitdata(x, y)
    o[:nclass] = length(unique(y))

    trials = Trials()
    best_args = fmin(objective,
    space=[quniform("num_round",o[:num_round_min], o[:num_round_max],o[:num_round_step]),
        quniform("max_depth",o[:max_depth_min], o[:max_depth_max],o[:max_depth_step]),
        uniform("eta", o[:η_min], o[:η_max])],
        algo=TPESUGGEST,
        maxevals=o[:maxevals],
        trials = trials)

    best_loss, best_ind = findmin(losses(trials))
    best_bst_file = trials["results"][best_ind]["bst_file"]
    best_space = trials["results"][best_ind]["space"]
    best_bst = Booster(model_file = best_bst_file)
    best_bst, best_args, best_space, best_loss
end

export predict
function predict(bst::XGBoost.Booster, x)
    x = x'
    I = XGBoost.predict(bst, x) # integer
    l = int2label(I) # label
end

end # end of module Boost
