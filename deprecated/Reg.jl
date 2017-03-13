"""
    include(Pkg.dir("Knet/examples/housing.jl"))
    (xtrn,ytrn,xtst,ytst) = Housing.loaddata()
    import Reg
    net, args, space, loss = Reg.main(xtrn, ytrn; gpu=true);
    preds = Reg.predict(net, xtrn)
    using Plots; p = scatter(ytrn', preds')
"""
module Reg

o = Dict()
o[:Lmin] = 1 # maximum number of layers
o[:Lmax] = 2 # maximum number of layers
o[:Nmin] = 50 # minimum number of neurons
o[:Nstep] = 50 # step of number of neurons
o[:Nmax] = 100 # maximum number of neurons
o[:epochs] = 10
o[:maxevals] = 10
o[:lr_min] = 0.01# minimum learning rate
o[:lr_max] = 0.5

using Knet, Hyperopt, JLD
export Knet,JLD

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end

function splitdata(x, y)
    n1, n2 = size(x)
    xtrn = x[:,1:4*n2÷5]
    xtst = x[:,4*n2÷5+1:n2]
    ytrn = y[:,1:4*n2÷5]
    ytst = y[:,4*n2÷5+1:n2]
    return  xtrn, ytrn, xtst, ytst
end

function preprocess(x,y)
    o[:Mx] = Mx = maximum(x,2)
    o[:mx] = mx = minimum(x,2)
    o[:My] = My = maximum(y,2)
    o[:my] = my = minimum(y,2)
    x = (x.- mx)./(Mx .- mx .+ 1e-20)
    y = (y.- my)./(My .- my .+ 1e-20)
    return x,y
end

function predict1(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = relu(x) # max(0,x)
        end
    end
    return x
end

function loss(w,x,y)
    ypred = predict1(w,x)
    sumabs2(y - ypred) / size(y,2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr=.5, epochs=10)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
    end
    return w
end

function err(w,dtrn)
    cost = ninstance = 0.0
    for (x, ygold) in dtrn
        cost += loss(w,x,ygold)
        ninstance += 1.0
    end
    cost/ninstance
end


function weights(data,h...; winit=0.1)
    x0,y0 = data[1]
    w = Any[]
    x = size(x0,1)
    for y in [h..., size(y0,1)]
        push!(w, convert(o[:atype], winit*randn(y,x)))
        push!(w, convert(o[:atype], zeros(y, 1)))
        x = y
    end
    return w
end

function minibatch(x, y, batchsize; atype=Array{Float32})
    x = atype(x); y = atype(y)
    data = Any[]
    for i=1:batchsize:size(x,2)
        j=min(i+batchsize-1,size(x,2))
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

function objective(args)
    h, lr = args
    h = Int.(collect(h))

    w = weights(o[:dtrn],h...)

    loss = 1000
    for i=1:o[:epochs]
        train(w, o[:dtrn]; lr=lr, epochs=1)
        cost = err(w, o[:dtst])
        if cost < loss
            loss = cost
        end
    end
    @printf("\nnlayer=%d,layers=%s,lr=%6.4f,loss=%6.4f\n",length(h),h,lr,loss)

    o[:bestfile] = tempname()*".jld"
    save(o[:bestfile], "net", w)
    o[:atype] == KnetArray{Float32} && Knet.knetgc()
    o[:atype] == Array{Float32} && Knet.gc()
    return Dict("loss" => loss, "status" => STATUS_OK,
    "net_file" => o[:bestfile], "space" => args)
end

export main
function main(x, y; batchsize=50, gpu=false)
    x, y = preprocess(x, y)
    xtrn, ytrn, xtst, ytst = splitdata(x, y)
    gpu == false ? (o[:atype] = Array{Float32}) : (o[:atype] = KnetArray{Float32})
    o[:dtrn] = minibatch(xtrn, ytrn, batchsize; atype=o[:atype])
    o[:dtst] = minibatch(xtst, ytst, batchsize; atype=o[:atype])

    trials = Trials()
    hps = [] # hyper-parameters
    opt = [] # option

    for l = o[:Lmin]:o[:Lmax]
        push!(hps,quniform("h$l",o[:Nmin],o[:Nmax],o[:Nstep]))
        push!(opt,hps)
    end

    best_args = fmin(objective,
    space=[choice("hidden",opt),uniform("lr", o[:lr_min], o[:lr_max])],
        algo=TPESUGGEST,
        maxevals=o[:maxevals],
        trials = trials)

    best_loss, best_ind = findmin(losses(trials))
    best_net_file = trials["results"][best_ind]["net_file"]
    best_space = trials["results"][best_ind]["space"]
    best_net = load(best_net_file,"net")
    best_net, best_args, best_space, best_loss
end



export predict
function predict(model::Array{Any}, x)
    x = (x .- o[:mx])./(o[:Mx] .- o[:mx] .+ 1e-20)
    y = Array(predict1(model, o[:atype](x)))
    y = o[:my] .+ (o[:My] .- o[:my]).*y
end


end# end of module Reg
