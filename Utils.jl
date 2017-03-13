module Utils
__precompile__()

using DataFrames
export DataFrame, by, dropna, rename

using Reexport
export @reexport

using Lazy: @as, @>, @>>
export @as, @>, @>>

using Parameters: @with_kw
export @with_kw

using StatsBase

@reexport using Logging

using PyCall
# @reexport using Plots
# @reexport using StatPlots

# import Base: +, -
# +(f::Function, a::Number) = x->(f(x) + a)
# -(f::Function) = x->-f(x)

export git
function git(path = pwd(), suffix = ".jl")
	folder = splitdir(path)[end]
	cd(path)
  run(`git config --global user.name "Yao Lu"`)
  run(`git config --global user.email "luyaocns@gmail.com"`)
	run(`git init`)
	try run(`git remote add $folder git@github.com:AStupidBear/$folder$suffix.git`) end
	try run(`git pull $folder master`) end
	run(`git add .`)
	run(`git commit -m "$(now())"`)
	run(`git push $folder master`)
	run(`git remote remove $folder`)
end

export typename
typename{T}(x::T) = string(T.name)

export proxy
function proxy(url)
  regKey = "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings"
  run(`powershell Set-ItemProperty -path \"$regKey\" AutoConfigURL -Value $url`)
end

cow() = proxy("http://127.0.0.1:7777/pac")


function linux_backup(dir = "/home/hdd1/YaoLu/Software", user = "luyao")
  date = string(now())[1:10]
  sysfile = joinpath(dir, "$date-sys.tar.gz")
  run(`sudo tar czf $file --exclude=/home --exclude=/media --exclude=/dev --exclude=/mnt --exclude=/proc --exclude=/sys --exclude=/tmp --exclude=/run /`)
  userfile = joinpath(dir, "$date-$user.tar")
  run(`sudo 7z a $userfile /home/$user`)
end

function linux_restore(file)
  run(`tar xf $(abspath(file)) -C /`)
end

function Base.shuffle(x::Array, y::Array)
  a = randperm(length(y))
  x, y = cget(x, a), cget(y, a)
end

export typeparam
typeparam{T}(x::T) = T.parameters[1]

export memory
memory(x) = Base.summarysize(x) / 1024^2

export cron
"""cron("spam.jl", 1)"""
function cron(fn, repeat)
  name, ext = splitfile(fn)
  vb = """
  DIM objShell
  set objShell=wscript.createObject("wscript.shell")
  iReturn=objShell.Run("cmd.exe /C $(abspath(fn))", 0, TRUE)
  """
  bat = """
  schtasks /create /tn "$name" /sc minute /mo $repeat /tr "$(abspath("$name.vbs"))"
  schtasks /run /tn "$name"
  """
  write("$name.vbs", vb)
  write("task.bat", bat)
  run(`task.bat`)
end



export @debug, @info
macro debug(ex)
  x = gensym()
  :($x = $ex; debug($x); $x) |> esc
end
macro info(ex)
  x = gensym()
  :($x = $ex; info($x); $x) |> esc
end

export @replace
macro replace(ex)
    ex = macroexpand(ex)
    typs = []
    names = [] # fieldnames
    for sym in ex.args[1].args[2:end]
        if isa(sym, Expr)
            push!(names, sym.args[1])
            push!(typs, sym.args[2])
        end
    end

    for (typ,n) in zip(typs, names)
        for f in fieldnames(eval(current_module(),typ))
            exreplace!(ex.args[2], :($f), :($n.$f))
        end
    end
    # @show ex
    esc(ex)
end


export @undict
"""
    d = Dict("a"=>[1,2,3], "b" => 2)
    @undict d a b
"""
macro undict(d, exs...)
  blk = Expr(:block)
  for ex in exs
    push!(blk.args, :($ex = $d[$(string(ex))]))
  end
  esc(blk)
end

"vcat(([1,2,3],[4,5,6]), ([1,2,3], [4,5,6]))"
function Base.vcat(X::Tuple...)
  ntuple(length(X[1])) do j
    mapreduce(i->X[i][j], vcat, 1:length(X))
  end
end

# function smotetomek(x, y)
#   SMOTETomek = pyimport("imblearn.combine")[:SMOTETomek]()
#   for t in 1:length(unique(y)) - 1
#     x, y = SMOTETomek[:fit_sample](x, y)
#   end
#   return x, y
# end
#
# function smote(x, y)
#   SMOTE = pyimport("imblearn.over_sampling")[:SMOTE]()
#   for t in 1:length(unique(y)) - 1
#     x, y = SMOTE[:fit_sample](x, y)
#   end
#   return x, y
# end
#
# function nearmiss(x, y)
#   NearMiss = pyimport("imblearn.under_sampling")[:NearMiss](
#     version = 2, random_state = 42)
#   NearMiss[:fit_sample](x, y)
# end
#
# function tomeklinks(x, y)
#   TomekLinks = pyimport("imblearn.under_sampling")[:TomekLinks]()
#   for t in 1:length(unique(y)) - 1
#     x, y = TomekLinks[:fit_sample](x, y)
#   end
#   return x, y
# end
#
# export balance
# """
#     x = rand(200, 3); y = rand(1:3, 200)
#     xb, yb = balance(x, y)
#
#     x = rand(3, 200); y = rand(1:3, 200)
#     xb, yb = balance(x, y)
# """
# function balance(x, y, method = smotetomek)
#   if size(x, 2) > size(x, 1)
#     xb, yb = method(x', vec(y))
#     return xb', yb
#   else
#     return method(x, vec(y))
#   end
# end

export balance
"""
    using Utils
    x = rand(4, 100)
    y = randprob(-1:1, [0.2, 0.6, 0.2], 100)
    xb, yb = balance(x, y)
    hist1(y, -1.5:1.5)
    hist1(yb, -1.5:1.5)
"""
function balance(x, y)
  xsize = size(x); ysize = size(y)
  xsize[1] > xsize[2] && (x = x')
  d = Dict()
  for i in eachindex(y)
    get!(d, y[i], [x[:, i]])
    push!(d[y[i]], x[:, i])
  end

  xb, yb = similar(x), similar(y)
  ny, key, vals = length(d), collect(keys(d)), collect(values(d))

  for i in 1:length(y)
    _ = rand(1:ny)
    yb[i] = key[_]
    xb[:, i] = rand(vals[_])
  end
  return reshape(xb, xsize), reshape(yb, ysize)
end


export readabsdir
readabsdir(dir) = map(file->joinpath(dir, file), readdir(dir))

export csize, clength, ccount, cview, cget, cset!, size2, size2
csize(a)=(ndims(a)==1 ? size(a) : size(a)[1:end-1])
csize(a,n)=tuple(csize(a)..., n) # size if you had n columns
clength(a)=(ndims(a)==1 ? length(a) : stride(a,ndims(a)))
ccount(a)=(ndims(a)==1 ? 1 : size(a,ndims(a)))
cview(a,i)=(ndims(a)==1 ? (@view a[i]) : view(a, ntuple(i->(:), ndims(a)-1)..., i))
cget(a,i)=(ndims(a)==1 ? a[i] : getindex(a, ntuple(i->(:), ndims(a)-1)..., i))
cset!(a,x,i)=(ndims(a)==1 ? (a[i] = x) : setindex!(a, x, ntuple(i->(:), ndims(a)-1)..., i))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd)))) # size as a matrix
size2(y,i)=size2(y)[i]


export hasnan
function hasnan(x)
  for i in eachindex(x)
    isnan(x[i]) && return true
  end
  false
end

function Base.readall(f::IO, T)
  x = Vector{T}()
  while !eof(f)
    push!(x, read(f, T))
  end
  x
end
function Base.readall(fn::AbstractString, T)
  x = Vector{T}()
  open(fn, "r") do f
    while !eof(f); push!(x, read(f, T));  end
  end
  x
end

export centralize, centralize!
"transform x to (-1, 1)"
function centralize!(x, dim=1)
  _max = maximum(x, dim)
  _min = minimum(x, dim)
  x .= (x .- _min) ./ (_max .- _min)
  x .= 2 .* x .- 1
end
centralize(x, dim=1) = centralize!(deepcopy(x), dim)

export splitfile
function splitfile(fn)::Tuple{String, String}
  _ = split(fn, '.')
  length(_) > 1 ? (_...) : (fn, "")
end

export tempfile
"""write(tempfile("hihi.txt"), "12")"""
function tempfile(fn)
  date = @> string(now())[1:16] replace(":", "-")
  name, ext = splitfile(fn)
  joinpath(tempdir(), date * "_" * name * "." *ext)
end

export @catch
macro catch(ex)
  quote
    try
      $ex
    catch e
      Logging.err(e)
    end
  end |> esc
end

###############################################################################
# BoundEncoder
###############################################################################
Base.rand(b::NTuple{2, Real}, dims...) = b[1] + rand(dims...) * (b[2] - b[1])

export purturb
function purturb(x, X, bounds)
  for j in 1:size(X, 2)
    if isapprox(x, X[:, j])
      for i in 1:length(x)
        x[i] += rand(bounds[i])
      end
    end
  end
  x
end

tobound(b) = b
tobound(b::Vector) = 1:length(b)

discretize(c::Range, x) = @as _ x indmin(abs2(_ .- c)) c[_]
discretize(c::NTuple{2, Number}, x) = x < c[1] ? c[1] :
                                    x > c[2] ? c[2] : x
discretize(c::Vector, x) = (i = discretize(1:length(c), x); c[i])

export BoundEncoder, transform

type BoundEncoder
  configs::Tuple
  bounds::Array{NTuple{2, Float64}}
end

function BoundEncoder(configs)
  bounds = []
  for c in configs
    length(c) > 1 && push!(bounds, tobound(c))
  end
  configs = tuple(configs...)
  bounds = [Float64.(extrema(b)) for b in bounds]
  BoundEncoder(configs, bounds)
end

function transform(encoder::BoundEncoder, x)
  i = 0; c = []
  for cc in encoder.configs
    push!(c, length(cc) > 1 ? (i += 1; discretize(cc, x[i])) : cc)
  end
  c
end

function inverse_transform(encoder::BoundEncoder, c)
  x = Float64[c[i] for i in eachindex(c) if length(encoder.configs[i]) > 1]
end

# configs = ((-1, 1), 1, 1:10, ["1", 3, 2])
# encoder = BoundEncoder(configs)
# x = [0, 5.3, 3]
# c = transform(encoder, x)
# @assert x == inverse_transform(encoder, c)

###############################################################################
# end of BoundEncoder
###############################################################################


export hist1
hist1(o...) = StatsBase.fit(Histogram, o...).weights

export @DataFrame
macro DataFrame(exs...)
  x = Expr(:call, :DataFrame)
  for ex in exs
    push!(x.args, Expr(:kw, ex, ex))
  end
  esc(x)
end

Base.Int64(x::Union{Float64, Float32}) = round(Int64, x)
Base.Int32(x::Union{Float64, Float32}) = round(Int32, x)


###############################################################################
# Parameter
###############################################################################
export Parameter
abstract Parameter

export getparam
getparam(param::Parameter) = fieldvalues(param)[1:(end - 1)]

export catparam
catparam{T<:Parameter}(param::T, c) = T(c..., param.bounds)

function Base.display(param::Parameter)
  println("Parameters: ")
  for s in fieldnames(param)
    s != :bounds && println(s, "=", getfield(param, s))
  end
end

export @rep
macro rep(n, ex)
  name = ex.args[1].args[1] |> string
  exs = [exreplace(ex, Symbol(name), Symbol(name*"$i")) for i in 1:n]
  @as _ exs Expr(:block, _...) esc
end

export @param
macro param(ex)
  ex = macroexpand(ex)
  bounds = []
  delete_line!(ex)
  unroll_block!(ex.args[3])
  args = ex.args[3].args
  for arg in args
    if arg.head != :line
      _ = arg.args[2]
      if isa(_, Expr)
        push!(bounds, _.args[2])
        arg.args[2] = _.args[1]
      else
        push!(bounds, _)
      end
    end
  end
  push!(args, :(bounds::Tuple=$(Expr(:tuple, bounds...))))
  esc(:(@with_kw $ex))
end

function Base.insert!{T}(collection::Array{T, 1}, index::Integer, items::Array{T, 1})
  for item in items
    Base.insert!(collection, index, item)
    index += 1
  end
  collection
end

export unroll_block!
unroll_block!(ex) = nothing
function unroll_block!(ex::Expr)
  args = ex.args
  i = 1
  while i <= length(args)
    if isa(args[i], Expr) && args[i].head == :block
      _ = deepcopy(args[i].args)
      deleteat!(args, i)
      insert!(args, i, _)
      i -= 1
    else
      unroll_block!(args[i])
    end
    i += 1
  end
end

export delete_line!
delete_line!(ex) = nothing
function delete_line!(ex::Expr)
  args = ex.args
  i = 1
  while i <= length(args)
    if isa(args[i], Expr) && args[i].head == :line
      deleteat!(args, i)
      i -= 1
    else
      delete_line!(args[i])
    end
    i += 1
  end
end


export unroll
unroll(x, name) = [getfield(x, s) for s in fieldnames(x) if
                  contains(string(s), string(name))]

###############################################################################
# end of Parameter
###############################################################################

export match
function Base.match(x, a::AbstractVector, b::AbstractVector)
  for i in eachindex(a)
    x == a[i] && return b[i]
  end
end

export rsqaured
rsqaured(ypred, y) = 1 - sum(abs2, ypred .- y) / sum(abs2, ypred .- mean(y))


###############################################################################
# Cloud
###############################################################################
function transfer_system(ip, user, port)
  rsync(homedir()*"/", "$user@$ip:~/", port)
  # rsync("/etc/apt/sources.list", "root@$ip:/etc/apt/sources.list", port)
  # ssh_eval = "sudo aptitude update && cat pkglist | xargs sudo aptitude install -y"
  # cmd = `ssh -p $port $user@$ip -t $ssh_eval`
  # run(cmd)
end

export ustc, highchain
ustc() = transfer_system("172.16.1.17", "luyao", 22)
highchain() = transfer_system("101.231.45.146", "luyao", 8822)

export linux_path, cygdrive
linux_path(path) = replace(path, "\\", "/")
cygdrive(path) = @> path linux_path replace(":", "") _->"/cygdrive/$_"

export junocloud
function junocloud(ip, user, port)
  local_root = joinpath(homedir(), "Documents", "Codes") |> linux_path
  remote_root = "/home/$user/Documents"

  m = match(r"connect\((.*)\)", clipboard())
  remoteport = is(m, nothing) ? "55440" : m[1]
  localport = "1234"

  julia_eval = """using Juno;Juno.connect(1234)"""
  ssh_eval = """chmod 400 ~/.ssh/id_rsa; ~/julia-0.5/bin/julia -i -e "$julia_eval"; bash"""

  src= joinpath(homedir(), ".juliarc.jl") |> cygdrive
  dst = "$user@$ip:/home/$user/.juliarc.jl"
  rsync(src, dst, port)

  src = joinpath(homedir(), ".ssh", "id_rsa") |> cygdrive
  dst = "$user@$ip:/home/$user/.ssh/id_rsa"
  rsync(src, dst, port)

  src = cygdrive(local_root)
  dst = "$user@$ip:$remote_root"
  rsync(src, dst, port)

  cmd = `ssh -X -R $localport:localhost:$remoteport -p $port
     $user@$ip -t $ssh_eval`
  run(cmd)
end
###############################################################################
# end of Cloud
###############################################################################

export rowvec, colvec
rowvec(x) = reshape(x, length(x), 1)
colvec(x) = reshape(x, 1, length(x))

export labelplot
function labelplot(x, label)
  p = Main.plot(; legend = nothing)
  t0 = 1
  for t in 2:length(x)
    if label[t] != label[t-1] || t == length(x)
      Main.plot!(p, t0:t, x[t0:t]; color=label[t-1], label=label[t-1])
      t0 = t
    end
  end
  return p
end

export imconvert
function imconvert(ext1, ext2)
  for (root, dirs, files) in walkdir(pwd())
    for file in files
      name, ext = splitfile(joinpath(root, file))
      if ext == ext1
        name1 = name * "." * ext1
        name2 = name * "." * ext2
        run(`imconvert $name1 $name2`)
      end
    end
  end
end

export fieldvalues
fieldvalues(x) = [getfield(x, s) for s in fieldnames(x)]

###############################################################################
# LabelEncoder
###############################################################################
export LabelEncoder, fit_transform, fit, transform, inverse_transform
@with_kw type LabelEncoder
  unique_label::Array = []
end

function fit_transform(encoder::LabelEncoder, label)
  fit(encoder, label)
  transform(encoder, label)
end

function fit(encoder::LabelEncoder, label)
  encoder.unique_label = unique(label)
  (eltype(encoder.unique_label) <: Number) && sort!(encoder.unique_label)
end

function transform(encoder::LabelEncoder, label)
  [findfirst(encoder.unique_label, l) for l in label]
end

function inverse_transform(encoder::LabelEncoder, index)
  [encoder.unique_label[i] for i in index]
end
###############################################################################
# end of LabelEncoder
###############################################################################

###############################################################################
# OneHotEncoder
###############################################################################
export OneHotEncoder, fit_transform, fit, transform, inverse_transform
"""
    label = [1, 3, 2]' #["small", "medium", "large"]'
    encoder = OneHotEncoder()
    vector = fit_transform(encoder, label)
    inverse_transform(encoder, vector) == label
"""
@with_kw type OneHotEncoder
  unique_label::Array = []
end

function fit_transform(encoder::OneHotEncoder, label)
  fit(encoder, label)
  transform(encoder, label)
end

function fit(encoder::OneHotEncoder, label)
  encoder.unique_label = unique(label)
  (eltype(encoder.unique_label) <: Number) && sort!(encoder.unique_label)
end

function transform(encoder::OneHotEncoder, label)
  vector = zeros(length(encoder.unique_label), length(label))
  for j in 1:size(vector, 2)
    i = findfirst(encoder.unique_label, label[j])
    vector[i, j] = 1
  end
  vector
end

function inverse_transform(encoder::OneHotEncoder, vector)
  label = Array{eltype(encoder.unique_label)}(1, size(vector, 2))
  for j = 1:size(vector, 2)
    i = indmax(vector[:, j])
    label[1, j] = encoder.unique_label[i]
  end
  label
end
###############################################################################
# end of OneHotEncoder
###############################################################################

###############################################################################
# MinMaxScaler
###############################################################################
export MinMaxScaler, fit_transform, fit, transform, inverse_transform
"""
    scaler = MinMaxScaler()
    x = [1 2 3 4; -1 -2 -3 -4]
    y = fit_transform(scaler, x)
    inverse_transform(scaler, y) == x
"""
@with_kw type MinMaxScaler
  sample_dim::Int = 2
  _max::Array{Float64} = []
  _min::Array{Float64} = []
end

function fit_transform(scaler::MinMaxScaler, x; dim = 2)
  fit(scaler, x; dim = dim)
  transform(scaler, x)
end

function fit(scaler::MinMaxScaler, x; dim = 2)
  scaler.sample_dim = dim
  scaler._max = maximum(x, 2)
  scaler._min = minimum(x, 2)
end

function transform(scaler::MinMaxScaler, x)
  y = (x .- scaler._min) ./ (scaler._max .- scaler._min .+ 1e-20)
end

function inverse_transform(scaler::MinMaxScaler, y)
  x = y .* (scaler._max .- scaler._min + 1e-20) .+ scaler._min
end
###############################################################################
# end of MinMaxScaler
###############################################################################
export ImageScaler, fit_transform, fit, transform, inverse_transform

"""
    scaler = ImageScaler()
    x = [1 2 3 4; -1 -2 -3 -4; 1 2 3 4; 2 2 3 4]
    fit_transform(scaler, x, (2, 2))
    fit_transform(scaler, x, (2, 2), false)
    fit(scaler, x, (2, 2))
    transform(scaler, x)
"""
@with_kw type ImageScaler
  scaler::MinMaxScaler = MinMaxScaler()
  imagesize::NTuple = ()
end

function fit_transform(scaler::ImageScaler, x, imagesize, reshape = true)
  fit(scaler, x, imagesize)
  transform(scaler, x, reshape)
end

function fit(scaler::ImageScaler, x, imagesize)
  scaler.imagesize = imagesize
  fit(scaler.scaler, x)
end

function transform(scaler::ImageScaler, x, reshape = true)
  xs = transform(scaler.scaler, x)
  reshape ? Base.reshape(xs, (scaler.imagesize..., ccount(x))) : xs
end

###############################################################################
# ImageScaler
###############################################################################
###############################################################################
# end of ImageScaler
###############################################################################

export softmax
softmax(x) = (y = exp(x); y ./ sum(y, 1))

export hardmax
hardmax(x) = x .== maximum(x, 1)
# function hardmax(x)
#   x = x .- maximum(x, 1)
#   1 + sign(x)
# end

macro curry(n, f)
    syms = [gensym() for i=1:n]
    foldl((ex, sym)  -> Expr(:->, sym, ex), Expr(:call, f, syms...), reverse(syms))
end

export partial
partial(f, a...) = ((b...) -> f(a...,b...))


export @print
macro print(expr...)
  str = join(["$ex = %.2f " for ex in expr], ", ")
  esc(:(@printf($str, $(expr...))))
end

Base.max(itr) = maximum(itr)
Base.min(itr) = minimum(itr)

export xcov
xcov(A,B) = xcorr(A .- mean(A), B .- mean(B))

"entropy(randn(10000))"
function entropy(x)
	edge, counts = histogram(x)
	P = counts / sum(counts)
	H = sum(-P .* log(2, P))
	scale = log(2, length(edge) - 1)
	return H / scale
end


"mutual_info(randn(10000), randn(10000))"
function mutual_info(x, y)
	ε = 1e-100
	edge1, edge2, counts = histogram2D(x, y)
	Pxy = counts / sum(counts)
	Px = sum(Pxy, 1)
	Py = sum(Pxy, 2)
	Hx = sum(-Px.*log(2, Px))
	Hy = sum(-Py.*log(2, Py))
	I = sum(Pxy .* log(2, (Pxy + ε) ./ (Py * Px + ε)))
	scale = log(2, (length(edge1) - 1) * (length(edge2) - 1))
	return Hx / scale, Hy / scale, I / scale
end

"kl_diverge(randn(100000), randn(100000) + 10)"
function kl_diverge(x1,x2)
	ε = 1e-100
  low = minimum([x1; x2])
	up = maximum([x1; x2])
	N = 1000
	edge = linspace(low,up,N+1)
	~, counts1 = hist(x1, edge)
	~, counts2 = hist(x2, edge)
	P1 = counts1 / sum(counts1)
	P2 = counts2 / sum(counts2)
	center = edge2center(edge)
	D_kl_diverge = sum(P1 .* log(2, P1 ./ (P2 + ε) + ε))
end

"edge2center(0:0.1:1)"
function edge2center(edge)
	dx = edge[2] - edge[1]
	center = edge[1:(end - 1)] + dx / 2
end

export randprob
"randprob([-1,1], [0.2,0.8], 10, 10)"
function randprob(list, prob)
  Z = 1.0 # partition number
  for i in eachindex(list)
    rand() < prob[i] / Z && return list[i]
    Z -= prob[i]
  end
end
function randprob(list, prob, dims...)
  x = zeros(dims...)
  for i in eachindex(x)
    x[i] = randprob(list, prob)
  end
  x
end


"using Distributions; rand(MvNormal([1.,2.,3.], [1.,1.,1.]), 1000)"
function pca(data::Array{Float64,2}, n=1)
  N = size(data, 2)
  data = data .- mean(data, 2)
  data_cov =1 / N * data * data.'
  D, V = eig(data_cov)
  idx = sortperm(D, rev=true)
  D = D[idx]
  V = V[:,idx]
  data_new = V.'[1:n,:] * data
  return D[1:n], V[:,1:n]
end

"""
	addprocs(3)
	T = 1000000;
	s1 = 2 * rand(T) - 1;	s2 = 2 * rand(T) - 1
	S0 = [s1'; s2'];
  X = S0 * [cos(π/4) sin(π/4); -sin(π/4) cos(π/4)]
	@time A = ICA(X, 2)
"""
function ica(X::Array{Float64,2}, M::Int64)
  X = X.- mean(X,2)
  N, T = size(X)
  Σ = 1 / T * X * X'
  d, U = eig(Σ)
  d = d[1:M]; U = U[:,1:M]
  Q = diagm(d.^(-1/2)) * U'
  Z = Q * X
  W = zeros(M, N)

  for k = 1:M
    w = randn(N); w = w / norm(w)
    Δ = 1.0
    while Δ > 1e-3
      w0 = w
      expectation = 1 / T * @parallel (+) for i=1:1:T
        dot(w,Z[:,i])^3*Z[:,i]
      end
      w = expectation - 3w
      if k >= 2
        w = w - W[1:k-1,:]' * W[1:k-1,:] * w
      end
      w = w / norm(w)
      Δ = 1 - abs(dot(w, w0))
    end
    W[k, :] = w'
  end
  return W * Q
end

heaviside(x) = 0.5 * (1 + sign(x))
delta(x, δx = 1e-3) = (heaviside(x + δx / 2) - heaviside(x - δx / 2)) / δx
interval(x, xmin, xmax) = x > xmax ? 0.0 : (x < xmin ? 0.0 : 1.0)

export minimums
function minimums(x)
  mins = similar(x)
  xmin = x[1]
  @inbounds for i = eachindex(x)
    x[i] < xmin && (xmin = x[i])
    mins[i] = xmin
  end
  mins
end

export maximums
function maximums(x)
  maxs = similar(x)
  xmax = x[1]
  @inbounds for i = eachindex(x)
    x[i] > xmax && (xmax = x[i])
    maxs[i] = xmax
  end
  maxs
end

export code2cmd
code2cmd(str) = @> str replace("\n","") replace("\"", "\\\"")

export run
Base.run(s::String) = run(`$(split(s))`)

export forward_diffn
function forward_diffn(x, n)
  dx = zeros(x)
  for t in 1:(length(x) - n)
    dx[t] = x[t + n] - x[t]
  end
  dx
end

export diffn
function diffn(x, n)
  dx = zeros(x)
  for t in n+1:length(x)
    dx[t] = x[t] - x[t-n]
  end
  dx
end

export vplot
vplot(ps::Vector) = vplot(ps...)
function vplot(ps...)
  N = length(ps)
  Main.plot(ps..., size = (600, 400N), layout = (N, 1), leg = false)
end

export ptp
function ptp(x)
  xmin, xmax = extrema(x)
  xmax - xmin
end

using Lazy: isexpr, rmlines, splitswitch
export @switch
macro switch(args...)
  test, exprs = splitswitch(args...)

  length(exprs) == 0 && return nothing
  length(exprs) == 1 && return esc(exprs[1])

  test_expr(test, val) =
  test == :_      ? val :
  has_symbol(test, :_) ? :(let _ = $val; $test; end) :
                    :($test==$val)

  thread(val, yes, no) = :($(test_expr(test, val)) ? $yes : $no)
  thread(val, yes) = thread(val, yes, :(error($"No match for $test in @switch")))
  thread(val, yes, rest...) = thread(val, yes, thread(rest...))

  esc(thread(exprs...))
end

export has_symbol
function has_symbol(ex::Expr, s)
  for i in 1:length(ex.args)
    has_symbol(ex.args[i], s) && return true
  end
  false
end
function has_symbol(ex, s)
  ex == s
end

export @repeat
macro repeat(n, ex)
  quote
    for _ in 1:$n
      $ex
    end
  end
end

function slow()
    a = 1.0
    for i in 1:10000
        for j in 1:10000
            a+=asinh(i+j)
        end
    end
    return a
end

export rosenbrock
function rosenbrock(x)
  x = collect(x)
  z = sum( 100*( x[2:end] .- x[1:end-1].^2 ).^2 .+ ( x[1:end-1] .- 1 ).^2 )
  Float64(z)
end

# fmin = -1.04
export branin
function branin(v)
    x, y = v
    x, y = 15x - 5, 15y
    res = 1/51.95 * ((y - 5.1*x^2 / (4*π^2) + 5x/π - 6)^2 + (10 -10/8π)cos(x) -44.81)
end

function quadratic_form(x, A, y)
  dot(x, A*y)
end

# export download
# function Base.download(url::AbstractString, filename::AbstractString)
#   dir, base  = splitdir(filename)
#   # run(`aria2c  --max-connection-per-server=8 $url -d $dir -o $base`)
#   run(`aria2c  --max-connection-per-server=8 --http-proxy="http://127.0.0.1:1080" $url -d $dir -o $base`)
#   filename
# end

function test()
	open(joinpath(homedir(),"test.txt"), "a") do f
		write(f, string(now()),"\n")
	end
end

macro success(ex)
	quote
		fail = true
		while fail
			try
				$(esc(ex))
				fail = false
			catch e
				println(e)
			end
		end
	end
end

export rsync
function rsync(src, dst, port = 22; delete = true)
  if delete == true
    run(`rsync -avzPh --delete-after -e "ssh -p $port" $src $dst`)
  else
    run(`rsync -avzPh -e "ssh -p $port" $src $dst`)
  end
  # run(`rsync -avuzPh --cvs-exclude --delete-after -e "ssh -p $port" $src $dst`)
  # all verpose update zip progress humanreadable
end

macro plots()
	ex = :(import Plots; Plots.gr())
	if isdefined(:IJulia)
		ex = Expr(:block, ex,
		:(Plots.default(size=(600,300),html_output_format="png")))
	end
	esc(ex)
end

function require()
	ENV["PYTHON"] = "C:/PortableSoftware/Scoop/apps/python/3.6.0/python.exe"
	ENV["JUPYTER"] = "C:/PortableSoftware/Scoop/apps/python/3.6.0/Scripts/jupyter-notebook.exe"

	pkgs = ["WinRPM"]
	urls = ["https://github.com/AStupidBear/WinRPM.jl.git"]
	for (pkg, url) in zip(pkgs, urls)
		Pkg.installed(pkg) == nothing && (Pkg.clone(url);Pkg.build(pkg))
	end

	pkgs = ["GR", "Plots", "Gtk", "PyCall", "IJulia", "BenchmarkTools"]
	for pkg in pkgs
		Pkg.installed(pkg) == nothing && Pkg.add(pkg)
	end
end

function build()
	pkgs = ["WinRPM"]
	for pkg in pkgs
		Pkg.build(pkg)
	end

	pkgs = ["GR", "Plots", "Gtk", "PyCall", "IJulia", "BenchmarkTools"]
	for pkg in pkgs
		Pkg.build(pkg)
	end
end

export ps
export @ps_str
function ps(str)
	file = tempname()*".ps1"
  write(file, str)
  run(`powershell $file`)
end
macro ps_str(str)
	ps(str)
end
# macro ps_str(str)
#   str = replace(str,"\\\$", "\$")
#   str = replace(str, "\n", ";")
#   "run(`powershell $str`)" |> parse
# end
export @bat_str
macro bat_str(str)
  file = tempname()*".bat"
  write(file, str)
  run(`$file`)
end

export @bash_str
macro bash_str(str)
  str = replace(str,"\\\$", "\$")
  file = tempname()*".bash"
  write(file, str)
  run(`bash $file`)
end

function center(edge)
  c = [edge[i+1]/2 + edge[i]/2 for i = 1:length(edge)-1]
end

export smooth
function smooth(x, n, dim::Integer=1)
    s = similar(x)
    Rpre = CartesianRange(size(x)[1:dim-1])
    Rpost = CartesianRange(size(x)[dim+1:end])
    _smooth!(s, x, n, Rpre, size(x, dim), Rpost)
end

@noinline function _smooth!(s, x, n, Rpre, N, Rpost)
    for Ipost in Rpost
        for i = 1:N
            ind = max(1, i-n):min(N, i+n)
            for Ipre in Rpre
                s[Ipre, i, Ipost] = sum(x[Ipre, i, Ipost] for i in ind) / length(ind)
            end
        end
    end
    s
end
# function smooth(x, n)
#   N = length(x)
#   y = zeros(x)
#   @inbounds for i in 1:N
#     ind = max(1, i-n):min(N, i+n)
#     y[i] = sum(x[i] for i in ind) / length(ind)
#   end
#   y
# end

function opt_truc(f::Function, n::Integer=1)
    s = 0 # sum of series
    a = f(n) # a[n]
    a_post = f(n+1) # a[n+1]
    while abs(a_post) < abs(a)
        s += a
        n += 1
        a = a_post
        a_post = f(n+1)
    end
    s
end

# using FastArrays
function Bernoulli(m::Integer)
  B = FastArray(0:m){Float64}(:)
  fill!(B, 0)
  B[0] = 1
  for n in 1:m
    B[n] = 1 - sum(binomial(n, k) * B[k] / (n-k+1) for k in 0:n-1)
  end
  B[m]
end

function generate(pkgname)
  pkgname="Documenter"
  path = Pkg.dir(pkgname,"docs","make.jl")
  run(`vim $path`)
  include(path)
end

"pf=piecewise(:x,:([x>0, x==0, x<0]),:([2*x,-1,-x]))"
function piecewise(x::Symbol,c::Expr,f::Expr)
  n=length(f.args)
  @assert n==length(c.args)
  @assert c.head==:vect
  @assert f.head==:vect
  vf=Vector{Function}(n)
  for i in 1:n
    vf[i]=@eval $x->$(f.args[i])
  end
  return @eval ($x)->($(vf)[findfirst($c)])($x)
end

export clear_records
function clear_records(obj)
  for (key, val) in obj.records
    empty!(val)
  end
end

export @evalat
macro evalat(m, ex)
  ex1 = parse(":($ex)")
  esc(:(eval($m, $ex1)))
end

function sp_A_mul_B!(y, rowptr, colptr, I, J, A, x)
  fill!(y, zero(eltype(y)))
  for col in 1:length(colptr)-1
    xc = x[col]
    @inbounds for s = colptr[col] : (colptr[col+1]-1)
        y[I[s]] += A[s]*xc
    end
  end
end

export exreplace!, exreplace
exreplace(ex::Expr, r, s) = (ex = deepcopy(ex); exreplace!(ex, r, s))
exreplace(ex, r, s) = ex == r ? s : ex
function exreplace!(ex::Expr, r, s)
  for i in 1:length(ex.args)
      ex.args[i] = exreplace(ex.args[i], r, s)
  end
  ex
end

# export @replace
# macro replace(ex)
#     ex = macroexpand(ex)
#     typs = []
#     names = [] # fieldnames
#     for sym in ex.args[1].args[2:end]
#         if isa(sym, Expr)
#             push!(names, sym.args[1])
#             push!(typs, sym.args[2])
#         end
#     end
#
#     for (typ,n) in zip(typs, names)
#         for f in fieldnames(eval(current_module(),typ))
#             exreplace!(ex.args[2], :($f), :($n.$f))
#         end
#     end
#     # @show ex
#     esc(ex)
# end
# macro replace(ex)
#     typs = []
#     names = [] # fieldnames
#     for sym in ex.args[1].args[2:end]
#         if isa(sym, Expr)
#             push!(names, sym.args[1])
#             push!(typs, sym.args[2])
#         end
#     end
#
#     str = string(ex.args[2])
#     for (typ,n) in zip(typs, names)
#         for f in fieldnames(eval(current_module(),typ))
#             str = replace(str,Regex("(?<=\\W)$f\\b"),"$n.$f")
#         end
#     end
#     ex.args[2] = parse(str)
#     @show ex
#     esc(ex)
# end

export Float
typealias Float Float64

export dsparse
function dsparse(A)
  colptr = A.colptr
  rowptr = A'.colptr
  I = rowvals(A)
  V = nonzeros(A)
  J = zeros(I)
  s = 1
  for j = 1:A.n, i in nzrange(A, j)
      J[s] = j; s += 1
  end
  rowptr, colptr, I, J, V
end

export record!
function record!(obj)
    for (key, val) in obj.records
        if isa(key, Tuple)
          sym, ind = key
          push!(val, getindex(getfield(obj, sym),ind))
        else
          push!(val, copy(getfield(obj, key)))
        end
    end
end

export monitor
function monitor(obj, keys)
    for key in keys
        if isa(key, Tuple)
          sym, ind = key
        else
          sym = key
        end
        typ = typeof(getfield(obj, sym))
        obj.records[key] = Vector{typ}()
    end
end
function monitor(objs::Array, keys)
  for obj in objs
    monitor(obj, keys)
  end
end


export @constant
macro constant(ex)
    ex_const = Expr(:block)
    for arg in ex.args
        arg_const = Expr(:const)
        if arg.head == :(=)
            push!(arg_const.args,arg)
            push!(ex_const.args,arg_const)
        else
          push!(ex_const.args,arg)
        end
    end
    esc(ex_const)
end
macro constant(typ, ex)
    ex_const = Expr(:block)
    for arg in ex.args
        arg_const = Expr(:const)
        if arg.head == :(=)
            arg.args[2] = :($typ($(arg.args[2])))
            push!(arg_const.args,arg)
            push!(ex_const.args,arg_const)
          else
            push!(ex_const.args,arg)
          end
    end
    esc(ex_const)
end

export @export_const
macro export_const(ex)
    ex_const = Expr(:block)
    for arg in ex.args
        arg_const = Expr(:const)
        if arg.head == :(=)
            push!(ex_const.args,:(export $(arg.args[1])))
            push!(arg_const.args,arg)
            push!(ex_const.args,arg_const)
        else
          push!(ex_const.args,arg)
        end
    end
    esc(ex_const)
end
macro export_const(typ, ex)
    ex_const = Expr(:block)
    for arg in ex.args
        arg_const = Expr(:const)
        if arg.head == :(=)
            push!(ex_const.args,:(export $(arg.args[1])))
            arg.args[2] = :($typ($(arg.args[2])))
            push!(arg_const.args,arg)
            push!(ex_const.args,arg_const)
          else
            push!(ex_const.args,arg)
          end
    end
    esc(ex_const)
end


function inbounds!(x, l, u)
  @inbounds for i in eachindex(x)
    if x[i] < l[i]
      x[i] = -x[i] + 2*l[i]
    elseif x[i] > u[i]
      x[i] = -x[i] +2*u[i]
    end
  end
end

macro clear()
  workspace()
  ex = quote
    include(joinpath(JULIA_HOME, "..", "etc", "julia", "juliarc.jl"))
    Plots = LastMain.Plots
  end
  esc(ex)
end
# for name in names(LastMain,true)
#     if string(name)[1] != '#' && isconst(LastMain, name) &&
#       name!=:Main && name!=:LastMain && name!=:clear
#       ex = :($name = LastMain.$name)
#       eval(current_module(), ex)
#     end
# end

export import_module
"""
    module foo
      bar(x) = x
    end
    import_module(foo,Main)
"""
function import_module(m1, m2=Main)
  for name in names(m1,true)
    if string(name)[1] != '#'
      ex = :($name = getfield($m1,Symbol($(string(name)))))
      eval(m2,ex)
    end
  end
end

export seq2stack
"""
    x = rand(3,1000); y = rand(100,1); z = rand(100,3)
    seq2stack(3, x); seq2stack(3, x, y);
    seq2stack(3, x, y, z)
"""
function seq2stack{T}(tstack::Int, seq::Array{T})
    stack = zeros(T, size(seq,1) * tstack,
            size(seq, 2) - tstack + 1)
    for t = tstack:size(seq,2)
        stack[:, t-tstack+1] = vec(seq[:, t-tstack+1:t])
    end
    return stack
end
function seq2stack{T}(tstack::Int, x::Array{T}, ys...)
  xstack = seq2stack(tstack, x)
  ystack = []
  for y in ys
    push!(ystack, y[tstack:end, :])
  end
  xstack, ystack...
end
# function seqstack{T}(tstack::Int, x::Array{T})
#   stack = zeros(T, csize(x)..., tstack, ccount(x)-tstack+1)
#   for t = tstack:ccount(x)
#       cset!(stack, cget(x, t-tstack+1:t), t-tstack+1)
#   end
#   return stack
# end


export splitdata
function splitdata(x, y)
    n1, n2 = size(x)
    xtrn = x[:, 1:4*n2÷5]
    xtst = x[:, 4*n2÷5+1:n2]
    ytrn = y[:, 1:4*n2÷5]
    ytst = y[:, 4*n2÷5+1:n2]
    return  xtrn, ytrn, xtst, ytst
end

export indmax
"indmax(x,y,z)"
function Base.indmax(xs...) # mimic max(xs...)
  eltype(xs[1])[indmax([x...]) for x in zip(xs...)]
end

"indmax(x,dim)"
function Base.indmax(x, dim::Int) # mimic maximum(x,dim)
  findmax(x, dim)[2]
end

export @cat
"""
    p = @cat begin
      plot(rand(10))
      plot(rand(100))
    end
    @cat [1,2] [1,2,3]
"""
macro cat(exs)
  expr = Expr(:block,:(p=[]))
  for ex in exs.args
    ex.head!=:line && push!(expr.args,:(push!(p, $(esc(ex)))))
  end
  expr
end
macro cat(exs...)
  expr = Expr(:block,:(p=[]))
  for ex in exs
    push!(expr.args,:(push!(p, $(esc(ex)))))
  end
  expr
end

export len
len = length

"conv(ones(3)/3, rand(100), :origin)"
function Base.conv(u, v, o=:origin)
  Base.conv(u, v)[1:length(v)]
end

"""
	A= ones(5, 5)
	B=[-1. -1 -1;
	   -1 +8  -1;
	   -1 -1 -1]/8
	C=conv2(A, B, :origin)
"""
function Base.conv2(A::Array{Float64,2},B::Array{Float64,2}, o=:origin)
		n1, n2=size(A)
		m1, m2=size(B)
		C = Base.conv2(A,B)
		return C[(m1 - 1):(m1 + n1 - 2), (m2 - 1):(m2 + n2 - 2)]
end

export @dict
"""
    a=1; b=2
    d = @dict(a, b)
"""
macro dict(exs...)
  expr = Expr(:block,:(d = Dict()))
  for ex in exs
    push!(expr.args,:(d[$(string(ex))] = $(esc(ex))))
    # push!(expr.args,:(d[$(QuoteNode(ex))] = $(esc(ex))))
  end
  push!(expr.args,:(d))
  expr
end

export undict
"""
    d=Dict(:a=>1,:b=>2)
    undict(d)
"""
function undict(d)
  for (key, val) in d
    eval(current_module(),:($(key)=$val))
  end
end

"""
    name = "/tmp/tmp1.txt"
    attach = ["/tmp/tmp1.txt","/tmp/tmp2.txt"]
    sendmail(name)
    sendmail(name, attach)
"""
function mail(name)
  spawn(pipeline(`cat $name`,`mail -s "Computation Results" luyaocns@gmail.com`))
end
function mail(name, attach)
  spawn(pipeline(`cat $name`,`mail -s "Computation Results" --attach=$attach luyaocns@gmail.com`))
end

export @save_output
macro save_output(ex)
    quote
        originalSTDOUT = STDOUT
        (outRead, outWrite) = redirect_stdout()
        $(esc(ex))
        close(outWrite)
        data = String(readavailable(outRead))
        close(outRead)
        redirect_stdout(originalSTDOUT)
        println(data)
        open("/tmp/temp.txt","w") do fh
            write(fh, "Subject: Terminal Email Send\n\n")
            write(fh, data)
        end
	      spawn(`bash /home/hdd1/YaoLu/Backup/sendmail.sh`)
    end
end

macro save(name,ex)
  name = string(name)
  quote
    open("/tmp/$($name).txt","a") do f
      println(f,$(string(ex)))
      println(f,$(esc(ex)))
    end
  end
end

export ndgrid
function ndgrid{T}(vs::AbstractVector{T}...; mat=false, rev=false)
    n = length(vs)
    sz = map(length, vs)
    if mat == false
      out = ntuple(i->Array{T}(prod(sz)), n)
    else
      out = ntuple(i->Array{T}(sz), n)
    end
    s = 1
    for i=1:n
        if rev==true
          a = out[i]::Array
        else
          a = out[n-i+1]::Array
        end
        v = vs[i]
        snext = s*sz[i]
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end

function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end


function brian2julia()
  str=clipboard()

  str=replace(str,"def","function")

  str=replace(str, "gamma", "γ")
  str=replace(str, "sigma", "σ")
  str=replace(str, "alpha", "α")
  str=replace(str, "beta", "β")
  str=replace(str, "eta", "η")
  str=replace(str, "rho", "ρ")

  str=replace(str,"\'","\"")
  str=replace(str,r"array\((.*)\)",s"\1")
  str=replace(str, "random.", "")
  str=replace(str, "np.", "")
  str=replace(str, "plt.", "")
  str=replace(str, ";", "")
  str=replace(str, "**", "^")
  str=replace(str, "*ms", "")
  str=replace(str, "/ms", "")
  str=replace(str, "*mV", "")
  str=replace(str, "/mV", "")
  str=replace(str, ": Hz", "")
  str=replace(str, ": 1", "")

  clipboard(str)
end

function httpserver(ip = "101.231.45.146", user = "luyao", port = 8822)
  ssh_eval = "cd /tmp; ~/miniconda2/bin/python -m SimpleHTTPServer 8123"
  run(`explorer http://localhost:8000` & `ssh -L 8000:localhost:8123 -p $port $user@$ip -t $ssh_eval`)
end

function rockyou()
  file = joinpath(tempdir(), "rockyou.txt")
  open(file, "w") do f
    start = CartesianIndex(ntuple(x->Int('a'), 5))
    stop = CartesianIndex(ntuple(x->Int('z'), 5))
    for I in CartesianRange(start, stop)
      for c in I.I
        write(f, Char(c))
  	  write(STDOUT, Char(c))
      end
      write(f, '\n')
  	write(STDOUT, '\n')
    end
  end
end

enable(pkg) = try run(`apm.cmd enable $pkg`) end
disable(pkg) = try run(`apm.cmd disable $pkg`) end

function toggle(env_name, pkgs)
  if !haskey(ENV, env_name)
    run(`setx $env_name 0`)
  elseif ENV[env_name] == "1"
    disable.(pkgs)
    run(`setx $env_name 0`)
  elseif ENV[env_name] == "0"
    enable.(pkgs)
    run(`setx $env_name 1`)
  end
end


# using Base.Threads
#
# function threadcall(f::Function, run_on_thread::Int, args...; kwargs...)
#     @assert run_on_thread!=1
#     rr = Future()
#
#     function inner()
#         function fun()
#             if Base.Threads.threadid()==1
#                 yield()
#             end
#             if Base.Threads.threadid()!=run_on_thread
#                 return
#             end
#
#             ret = f(args...;kwargs...)
#             put!(rr, ret)
#         end
#         ccall(:jl_threading_run, Void, (Any,), Core.svec(fun))  #Run it on all threads
#         rr
#     end
#     @async inner()
#     rr
# end

# export spsum2!
# function spsum2!(W, fire, g)
#   j=1
#   spike = fire[j]
#   for i in eachindex(W.nzval)
#     if j < W.n && i > W.colptr[j+1]-1; j += 1; spike = fire[j];  end
#     if spike; g[W.rowval[i]] += W.nzval[i]; end
#   end
# end

# export @export_all
# macro export_all(ex)
#   ex_exp = Expr(:block)
#   for arg in ex.args
#       if arg.head == :(=)
#           push!(ex_exp.args,:(export $(arg.args[1])))
#       end
#       push!(ex_exp.args,arg)
#   end
#   esc(ex_exp)
# end


# export @fast
# macro fast(ex)
#   :(@fastmath(@inbounds($ex)))
# end

# export @param
# macro param(ex)
#   bounds = []
#   for arg in ex.args[3].args
#     if arg.head != :line
#       arg1 = arg.args[2]
#       if isdefined(arg1, :args)
#         push!(bounds, arg1.args[2])
#         arg.args[2] = arg1.args[1]
#       else
#         push!(bounds, arg1)
#       end
#     end
#   end
#   push!(ex.args[3].args, :(bounds::Tuple=$(Expr(:tuple, bounds...))))
#   esc(:(@with_kw $ex))
# end

# function histy(y)
#     y = vec2y(y)
#     if !isdefined(:nclass)
#         @eval nclass = $(Int(maximum(y))+1)
#     end
#     freq =  hist(vec(y),-0.5:1:nclass-0.5)[2]
#     prob = freq/sum(freq)
# end
# using Gadfly, xFrames
# function imbalance_view(train_Y, test_Y, preds)
#     xticks = Guide.xticks(ticks = [0:nclass-1;])
#     yticks = Guide.yticks(ticks = [0:nclass-1;])
#     df = xFrame(
#         class = test_Y,
#         predict = preds
#     )
#     plot(df, x = :class, y = :predict, Geom.violin, xticks, yticks,
#     Guide.title("Error Violin")) |> display
#     df = xFrame(
#         class = 0:nclass-1,
#         percent = histy(test_Y[test_Y .!= preds])
#     )
#     plot(df, x = :class, y = :percent, Geom.bar, xticks,
#     Guide.title("Error Contribution")) |> display
#     df_trn = xFrame(
#         class = 0:nclass-1,
#         percent = histy(train_Y),
#         group = "train"
#     )
#     df_tst = xFrame(
#         class = 0:nclass-1,
#         percent = histy(test_Y),
#         group = "test"
#     )
#     df_prd = xFrame(
#         class = 0:nclass-1,
#         percent = histy(preds),
#         group = "predict"
#     )
#     df = vcat(df_trn, df_tst, df_prd)
#     plot(df, x = :class, y = :percent, color = :group,
#     Geom.bar(position=:dodge), xticks,
#     Guide.title("Distribution")) |> display
# end


# module PlotUtils
#
# export imshow
# "imshow(rand(100,100))"
# function imshow(A)
#     @eval using Images,Colors
#     A = (A - minimum(A))./(maximum(A) - minimum(A))
#     convert(Image{Gray},A)
# end
#
# """
#     using Gadfly
#     mplot(x=1:10,y=rand(10,3),Gadfly.Geom.line)
# """
# function mplot(o1...;x=[], y=[], o2...)
#     @eval using DataFrames
#     df = DataFrame()
#     for j = 1:size(y,2)
#         df = vcat(df, DataFrame(x=x,y=y[:,j],label="$j"))
#     end
#     Gadfly.plot(df, x=:x, y=:y, color=:label,o1...,o2...) |> display
# end
#
# end

end # End of Utils
