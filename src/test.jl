using Utils
src = "/cygdrive/C/Users/AStupidBear/Documents/Codes"
dst = "luyao@60.210.253.206:/cygdrive/C/Users/luyao/Documents/"
rsync(src, dst)

function git(path = pwd(), suffix = ".jl")
	folder = splitdir(path)[end]
	cd(path)
  run(`git config --global user.name "Yao Lu"`)
  run(`git config --global user.email "luyaocns@gmail.com"`)
	run(`git init`)
  @show str = `git remote add $folder git@github.com:AStupidBear/$folder$suffix.git`
	try run(str) end
	try run(`git pull $folder master`) end
	run(`git add .`)
	try run(`git commit -m $(now())`) end
	run(`git push $folder master`)
	run(`git remote remove $folder`)
end

using Plots
using StatPlots
using PlotRecipes
# @step gcd(1,2)
#
# using Utils
# @step cygdrive("c:/users")

# code = """
# Set WinScriptHost = CreateObject("WScript.Shell")
# WinScriptHost.Run Chr(34) & "$(abspath(file))" & Chr(34), 0
# Set WinScriptHost = Nothing
# """



# function balance(x, y; vector = false, imblearn = false, normalize = false)
#     if normalize == true
#         xmin = minimum(x,ndims(x))
#         xmax = maximum(x,ndims(x))
#         x = (x .- xmin) ./ (xmax .- xmin)
#     end
#     y =  vec2y(y)
#     train_X = cget(x,1:4*ccount(x)÷5)
#     train_Y = y[:,1:4*end÷5]
#     test_X  = cget(x,4*ccount(x)÷5+1:ccount(x))
#     test_Y  = y[:,4*end÷5+1:end]
#     if imblearn == :SMOTE
#         train_X = reshape(train_X,size2(train_X))
#         @repeat 3 begin
#         SM = os.SMOTE(ratio = 1.0)
#         smx, smy = SM[:fit_sample](transpose(train_X), vec(train_Y))
#         train_X = transpose(smx); train_Y = transpose(smy)
#         end
#         train_X = reshape(train_X,(csize(x)...,ccount(train_X)))
#     elseif imblearn == :SMOTETomek
#         train_X = reshape(train_X,size2(train_X))
#         @repeat 3 begin
#         STK = cb.SMOTETomek(ratio = 1.0)
#         stkx, stky = STK[:fit_sample](transpose(train_X), vec(train_Y))
#         train_X = transpose(stkx); train_Y = transpose(stky)
#         end
#         train_X = reshape(train_X,(csize(x)...,ccount(train_X)))
#     end
#     if vector == true
#         train_Y = y2vec(train_Y)
#         test_Y = y2vec(test_Y)
#     end
#     return train_X, train_Y, test_X, test_Y
# end







# addprocs(3)
# using Utils, MLxsets; reload("MLP")
#
# x, y = MNIST.trainx()
# x = reshape(x, 28*28, 60000)
# x = x[1:10, 1:200]; y = y[1:200]
#
# net = MLP.Classifier(; epochs=1, maxevals=3)
# MLP.train!(net, x, y)
# MLP.train!(net, x, y; method = :Hyperband)
# MLP.train!(net, x, y; method = :BO)
# MLP.train!(net, x, y; method = :TPE)
# preds, probs = MLP.predict(net, x)

# 1
# Int(0.1)
# using Utils
# Int(0.1)
# using Utils, MLxsets
# reload("MLP")
#
# x, y = MNIST.trainx()
# x = reshape(x, 28*28, 60000)
# # x = x[1:10, 1:200]; y = y[1:200]
# net = MLP.Classifier()
#
# MLP.train!(net, x, y)
# # MLP.train_policy(net, x, y)
# preds, probs = MLP.predict(net, x)
#
# MLP.zeroone(net.w, net.dtst, MLP.mlp)

# using Knet
# include("hyperband.jl")
# hyperband(getconfig1, getloss1)

#
# function foo{T}(x)
#   @show T
#   # T(x)
# end
# foo{sin}(1)
#
# 1


# p = [0.4, 0.33, 1-0.4-0.32]#
# hardmax(x, n = 30) = (y = exp(n * x); y ./ sum(y, 1))
# hardmax(p, 30)
#
# using Knet
# Knet.logsumexp
#
#
# using Utils, MLxsets
# reload("MLP")
#
# x, y = MNIST.trainx()
# x = reshape(x, 28*28, 60000)
# x = x[1:10, 1:200]; y = y[1:200]
#
# net = MLP.Classifier(;nepochs=1, maxevals=1)
# MLP.train(net, x, y)
# MLP.train_policy(net, x, y)
# preds, probs = MLP.predict(net, x)


# using Utils: branin
# using Hyperopt
# x_min, y_min = Hyperopt.optimize(branin, 0:1e-2:1, 0:1e-2:1)
# foo(x) = x[1] + x[2] + x[3]
# space, trials= Hyperopt.optimize(foo, 0:1e-2:1, 0:1e-2:10, (0, 1))
#
# collect(space)
# # preds, probs = MLP.predict(clf, x)
#

# reload("Hyperopt")
#
# type foo
#   a::Int
# end
#
# objective(x; a=foo(1)) = (@show a.a; a.a=10; x + a.a)
# a = foo(1)
#
# f = Hyperopt.partial(objective; a=foo(1))
# trials = Hyperopt.Trials()
#
# Hyperopt.fmin(f, Hyperopt.uniform("x", -2, 2),
#                 algo=Hyperopt.TPESUGGEST, maxevals=10, trials=trials)
#
#
# Hyperopt.valswithlosses(trials)
