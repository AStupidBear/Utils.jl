"""
    text = readstring("/home/rluser/.julia/v0.5/Knet/data/100.txt")[1:30000]
    vocab = Dict{Char,Int}()
    for t in text, c in t; get!(vocab, c, 1+length(vocab)); end
    x = RNN.label2vec(text[1:end-1])
    y = text[2:end]
    RNN.save("/tmp/data.jld","x",x,"y",y,"vocab",vocab,"lu",RNN.o[:lu])

    import RNN
    RNN.@load("/tmp/data.jld")
    RNN.o[:lu] = lu

    RNN.o[:maxevals] = 1
    RNN.o[:Nmax] = RNN.o[:Nmin] = 256
    RNN.o[:Nstep] = RNN.o[:Lmin] = RNN.o[:Lmax] = 1
    RNN.o[:fast] = false

    import Utils
    Utils.@save_output begin
        net, args, space, loss = RNN.main(x,y)
        RNN.generate(net, vocab, 100)
        preds, probs, state = RNN.predict(net, space, x)
    end
"""

module RNN

o = Dict()
o[:embed] = 1
o[:epochs] = 30
o[:batchsize] = 50
o[:seqlength] = 100
o[:decay] = 0.9
o[:lr] = 1.0
o[:gclip] = 3.0
o[:winit] = 0.3
o[:gcheck] = 0
o[:seed] = -1
o[:atype] = Array{Float32}
o[:fast] = true

o[:Lmin] = 1 # minimum number of hidden layers
o[:Lmax] = 2 # maximum number of hidden layers
o[:Nmin] = 50  # minimum number of neurons
o[:Nstep] = 50 # N = Nmin:Nstep:Nmax
o[:Nmax] = 300 # maximum number of neurons

o[:lr_min] = 0.01 # minimum learning rate
o[:lr_max] = 0.5 # maximum learning rate
o[:maxevals] = 3 # maximum evaluations of hyperopt
############################################################

using Knet,AutoGrad,JLD,Hyperopt
export Knet,JLD

initialize(data, o) =
initweights(o[:atype], o[:hidden], data, o[:embed], o[:winit])

function minibatch(x, y, batch_size)
    n, dx, dy = size(x,2), size(x,1), size(y,1)
    nbatch = div(n, batch_size)
    data = [(zeros(batch_size,dx),zeros(batch_size,dy)) for i=1:nbatch ]
    cidx = 0
    for idx = 1:size(x,2)           # safest way to iterate over utf-8 text
        idata = 1 + idx % nbatch
        row = 1 + div(idx, nbatch)
        row > batch_size && break
        data[idata][1][row,:] = x[:,idx]
        data[idata][2][row,:] = y[:,idx]
    end
    return data
end

function train!(net, data, o)
    o[:bestfile] = tempname()*".jld"
    save(o[:bestfile], "net", net)
    s0 = initstate(o[:atype], o[:hidden], o[:batchsize])
    lr = o[:lr]
    if o[:fast]
        @time (for epoch=1:o[:epochs]
               train1(net, copy(s0), data[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
               end; Knet.cudaDeviceSynchronize())
        save(o[:bestfile], "net", net)
        acc = accuracy(net,copy(s0),data[1])
        return 1-acc, o[:bestfile]
    end
    losses = map(d->loss(net,copy(s0),d), data)
    println((:epoch,0,:loss,losses...))
    devset = ifelse(length(data) > 1, 2, 1)
    devlast = devbest = losses[devset]
    for epoch=1:o[:epochs]
        @time train1(net, copy(s0), data[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
        @time losses = map(d->loss(net,copy(s0),d), data)
        println((:epoch,epoch,:loss,losses...))
        if o[:gcheck] > 0
            gradcheck(loss, net, copy(s0), data[1], 1:o[:seqlength]; gcheck=o[:gcheck])
        end
        devloss = losses[devset]
        if devloss < devbest
            devbest = devloss
            info("Saving best net to $(o[:bestfile])")
            save(o[:bestfile], "net", net)
        end
        if devloss > devlast
            lr *= o[:decay]
            info("New learning rate: $lr")
        end
        devlast = devloss
    end
    acc = accuracy(net,copy(s0),data[devset])
    1-acc, o[:bestfile]
end


# sequence[t]: input token at time t
# state is modified in place
function train1(param, state, sequence; slen=100, lr=1.0, gclip=0.0)
    for t = 1:slen:length(sequence)-slen
        range = t:t+slen-1
        gloss = lossgradient(param, state, sequence, range)
        gscale = lr
        if gclip > 0
            gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
            if gnorm > gclip
                gscale *= gclip / gnorm
            end
        end
        gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
        for k in 1:length(param)
            # param[k] -= gscale * gloss[k]
            Knet.axpy!(-gscale, gloss[k], param[k])
        end
        isa(state,Vector{Any}) || error("State should not be Boxed.")
        # The following is needed in case AutoGrad boxes state values during gradient calculation
        for i = 1:length(state)
            state[i] = AutoGrad.getval(state[i])
        end
    end
end

# param[2k-1,2k]: weight and bias for the k'th lstm layer
# param[end-2]: embedding matrix
# param[end-1,end]: weight and bias for final prediction
function initweights(atype, hidden, data, embed, winit)
    dx, dy = size(data[1][1][1],2), size(data[1][1][2],2)
    param = Array(Any, 2*length(hidden)+3)
    input = embed
    for k = 1:length(hidden)
        param[2k-1] = winit*randn(input+hidden[k], 4*hidden[k])
        param[2k]   = zeros(1, 4*hidden[k])
        param[2k][1:hidden[k]] = 1 # forget gate bias
        input = hidden[k]
    end
    param[end-2] = winit*randn(dx,embed)
    param[end-1] = winit*randn(hidden[end],dy)
    param[end] = zeros(1,dy)
    return map(p->convert(atype,p), param)
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2*length(hidden))
    for k = 1:length(hidden)
        state[2k-1] = zeros(batchsize,hidden[k])
        state[2k] = zeros(batchsize,hidden[k])
    end
    return map(s->convert(atype,s), state)
end

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# s[2k-1,2k]: hidden and cell for the k'th lstm layer
# w[2k-1,2k]: weight and bias for k'th lstm layer
# w[end-2]: embedding matrix
# w[end-1,end]: weight and bias for final prediction
# state is modified in place
function predict1(w, s, x)
    x = x * w[end-2]
    for i = 1:2:length(s)
        (s[i],s[i+1]) = lstm(w[i],w[i+1],s[i],s[i+1],x)
        x = s[i]
    end
    return x * w[end-1] .+ w[end]
end

# sequence[t]: input token at time t
# state is modified in place
function loss(param,state,sequence,range=1:length(sequence)-1)
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(param[1]))
    for t in range
        input = convert(atype,sequence[t][1])
        ypred = predict1(param,state,input)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        ygold = convert(atype,sequence[t][2])
        total += sum(ygold .* ynorm)
        count += size(ygold,1)
    end
    return -total / count
end

function accuracy(param,state,sequence,range=1:length(sequence)-1)
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(param[1]))
    for t in range
        input = convert(atype,sequence[t][1])
        ypred = predict1(param,state,input)
        ygold = convert(atype,sequence[t][2])
        total += sum(ygold .* (ypred .== maximum(ypred,2)))
        count += size(ygold,1)
    end
    return total / count
end

lossgradient = grad(loss)

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end
#####################################################

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
    lu = o[:lu] # unique label
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

function generate(param, vocab, nchar)
    state = initstate(o[:atype], o[:hidden], 1)
    index_to_char = Array(Char, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end
    input = oftype(param[1], zeros(1,length(vocab)))
    index = 1
    for t in 1:nchar
        ypred = predict1(param,state,input)
        input[index] = 0
        index = sample(exp(logp(ypred)))
        print(index_to_char[index])
        input[index] = 1
    end
    println()
end

function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

function objective(args)
    h, lr = args
    h = Int.(collect(h))
    o[:hidden],o[:lr] = h,lr

    net = initialize(o[:data], o)
    loss, file = train!(net, o[:data], o)

    @printf("\nnlayer=%d,layers=%s,lr=%6.4f,loss=%6.4f\n",
    length(h),h,lr,loss)

    o[:atype] == KnetArray{Float32} && Knet.knetgc()
    o[:atype] == Array{Float32} && Knet.gc()
    return Dict("loss" => loss, "status" => STATUS_OK,
    "net_file" => file, "space" => args)
end

function main(x, y; gpu=false)
    gpu == true && (o[:atype] = KnetArray{Float32})
    x, y = preprocess(x, y)
    xtrn, ytrn, xtst, ytst = splitdata(x, y)
    dtrn = minibatch(xtrn, ytrn, o[:batchsize])
    dtst = minibatch(xtst, ytst, o[:batchsize])
    o[:data] = [dtrn, dtst]

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

function predict(net, state::Array, x)
    x = (x.- o[:m])./(o[:M] .- o[:m] .+ 1e-20)
    preds = Array{Any}(size(x,2))
    probs = zeros(size(x,2))
    for t = 1:size(x,2)
        y = Array(predict1(net, state, o[:atype](x[:,t]')))'
        probs[t] = maximum(exp(y)./sum(exp(y)))
        preds[t] = vec2label(y)[1]
    end
    return preds,probs,state
end
function predict(net, space::Tuple, x)
    h, lr = space
    h = Int.(collect(h))
    o[:hidden],o[:lr] = h,lr

    state = initstate(o[:atype], o[:hidden], 1)
    predict(net, state, x)
end

end # end of module RNN
