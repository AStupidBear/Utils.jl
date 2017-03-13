"""
    import CNN, MNIST
    reload("CNN")
    x,y = MNIST.traindata()
    # x = x[:,1:1000]; y = y[1:1000]
    net, args, space, loss = CNN.main(x,y,(28,28,1))
    preds, probs = CNN.predict(net, x[:,1:10])
"""
module CNN

o = Dict()
o[:Lmin] = 1 # maximum number of layers
o[:Lmax] = 2 # maximum number of layers
o[:Nmin] = 50 # minimum number of neurons
o[:Nstep] = 50 # step of number of neurons
o[:Nmax] = 1000 # maximum number of neurons
o[:Cmin] = 20  # minimum number of channels
o[:Cstep] = 10 # step of number of channels
o[:Cmax] = 50 # maximum number of channels
o[:Wmin] = 5   # minimum  window size
o[:Wstep] = 1  # step of  window size
o[:Wmax] = 5  # maximum window size
o[:epochs] = 4 # training epochs
o[:maxevals] = 3 # minimum evaluations of optimization functions
o[:lr_min] = 0.01 # minimum learning rate
o[:lr_max] = 0.5 # maximum learning rate

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

function label2vec(l)
    if !haskey(o,:lu)
      o[:lu] = unique(l)
      eltype(o[:lu])<:Number && (o[:lu] = sort(o[:lu]))
    end
    lu = o[:lu]
    eltype(lu)<:Number && (lu = sort(lu))
    v = zeros(length(lu),length(l))
    for j = 1:size(v,2)
        i = find(lu .== l[j])[1]
        v[i,j] = 1
    end
    v
end

function vec2label(v)
    lu = o[:lu]
    l = Array{eltype(lu)}(1,size(v,2))
    for j = 1:size(v,2)
        i = findmax(v[:,j])[2]
        l[1,j] = lu[i]
    end
    l
end

function preprocess(x,y)
    o[:M] = M = maximum(x,2)
    o[:m] = m = minimum(x,2)
    x = (x.- m)./(M .- m .+ 1e-20)
    y = label2vec(y)
    return x,y
end

function xavier(a...)
    w = rand(a...)
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w))
        fanin = div(length(w), fanout)
    end
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end

function weights(data,W,O,H)
    # W:windowsizes, O:outputchannels, H:hiddenlayers
    atype = o[:atype]
    x, y = atype(data[1][1]), atype(data[1][2])
    w = [Any[],Any[]]

    I = zeros(O) # input channel
    I[1] = size(x,3)
    I[2:end] = O[1:end-1]
    for i = 1:length(W)
        push!(w[1], atype(xavier(Float32,W[i],W[i],I[i],O[i])))
        x = conv4(w[1][end],x)
        push!(w[1], atype(zeros(Float32,1,1,O[i],1)))
        x =  pool(relu(x.+ w[1][end]))
    end
    x = mat(x)

    push!(w[2], atype(xavier(Float32,H[1],size(x,1))))
    push!(w[2], atype(zeros(Float32,H[1],1)))
    for i = 2:length(H)
        push!(w[2], atype(xavier(Float32,H[i],H[i-1])))
        push!(w[2], atype(zeros(Float32,H[i],1)))
    end
    push!(w[2], atype(xavier(Float32,size(y,1),H[end])))
    push!(w[2], atype(zeros(Float32,size(y,1),1)))
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

function minibatch4(x, y, batchsize, imagesize; atype=KnetArray{Float32})
    data = minibatch(x,y,batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (imagesize...,batchsize)), y)
    end
    return data
end

function predict1(w,x)
    for i = 1:2:length(w[1])
        x = pool(relu(conv4(w[1][i],x;padding=0) .+ w[1][i+1]))
    end
    x = mat(x)
    for i = 1:2:length(w[2])-2
        x = relu(w[2][i]*x .+ w[2][i+1])
    end
    return w[2][end-1]*x .+ w[2][end]
end

function accuracy(w, dtst)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        ypred = predict1(w, x)
        ynorm = ypred .- log(sum(exp(ypred),1))
        nloss += -sum(ygold .* ynorm)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (ygold .== maximum(ygold,1)))
        ninstance += size(ygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

function loss(w,x,ygold)
    x = convert(o[:atype], x)
    ygold = convert(o[:atype], ygold)
    ypred = predict1(w,x)
    ynorm = logp(ypred,1) # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr=.5, epochs=10)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
            for i in 1:length(w[1])
                w[1][i] -= lr * g[1][i]
            end
            for i in 1:length(w[2])
                w[2][i] -= lr * g[2][i]
            end
        end
    end
    return w
end


function objective(args)
    H, C, lr = args
    H = Int.(collect(H))
    W = Int[x[1] for x in C]
    O = Int[x[2] for x in C]

    w = weights(o[:dtrn], W, O, H)

    loss = 1000
    for i=1:o[:epochs]
        train(w, o[:dtrn]; lr=lr, epochs=1)
        err = 1 - accuracy(w, o[:dtst])[1]
        if err < loss
            loss = err
        end
    end
    @printf("\nWindowSize=%s,OutputChannels=%s,HiddenLayers=%s,lr=%6.4f,loss=%6.4f\n",
    W,O,H,lr,loss)

    o[:bestfile] = tempname()*".jld"
    save(o[:bestfile], "net", w)
    o[:atype] == KnetArray{Float32} && Knet.knetgc()
    o[:atype] == Array{Float32} && Knet.gc()
    return Dict("loss" => loss, "status" => STATUS_OK,
    "net_file" => o[:bestfile], "space" => args)
end

export main
function main(x, y, imagesize; batchsize=50, gpu=true)
    x, y = preprocess(x, y)
    xtrn, ytrn, xtst, ytst = splitdata(x, y)
    gpu == false ? (o[:atype] = Array{Float32}) : (o[:atype] = KnetArray{Float32})
    o[:imagesize] = imagesize
    o[:dtrn] = minibatch4(xtrn, ytrn, batchsize, imagesize; atype=o[:atype])
    o[:dtst] = minibatch4(xtst, ytst, batchsize, imagesize; atype=o[:atype])

    trials = Trials()
    hps_H = [] # hyper-parameters of H(idden)
    opt_H = [] # option of H(idden)
    hps_C = [] # hyper-parameters of C(onv) = (W,O)
    opt_C = [] # option of C(onv) =(W,O)

    for l = o[:Lmin]:o[:Lmax]
        push!(hps_H,quniform("H$l",o[:Nmin],o[:Nmax],o[:Nstep]))
        push!(opt_H,hps_H)
        push!(hps_C,(quniform("W$l",o[:Wmin],o[:Wmax],o[:Wstep]),quniform("O$l",o[:Cmin],o[:Cmax],o[:Cstep])))
        push!(opt_C,hps_C)
    end

    best_args = fmin(objective,
    space=[choice("hidden",opt_H), choice("conv",opt_C), uniform("lr", o[:lr_min], o[:lr_max])],
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
function predict(net, x)
    x = (x.- o[:m])./(o[:M] .- o[:m] .+ 1e-20)
    y = Array(predict1(net, o[:atype](reshape(x, (o[:imagesize]...,size(x,2))))))
    probs = exp(y)./sum(exp(y),1)
    preds = vec2label(y)
    return preds,probs
end

end# end of module CNN
