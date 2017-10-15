__precompile__()

module Utils

###############################################################################
# begin of load packages
###############################################################################
using Reexport; export @reexport

@reexport using Suppressor

# @reexport using BenchmarkTools

# @reexport using ClobberingReload

import StatsBase

using Lazy: @as, @>, @>>; export @as, @>, @>>

using Glob; export glob

@reexport using Polynomials

@reexport using Parameters

@reexport using MacroTools

@reexport using NamedTuples

@reexport using TypedTables

using DataFrames; export DataFrames, DataFrame
# export DataFrame, aggregate, describe, by, combine, groupby, nullable!, readtable, rename!, rename, tail, writetable, dropna, columns

using MLKernels

@reexport using DataStructures
###############################################################################
# end of load packages
###############################################################################

###############################################################################
# begin of kernel
###############################################################################
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

###############################################################################
# end of kernel
###############################################################################

grad1(x) = [mapslices(diff, x, dim) for dim in 1:ndims(x)]

grad2(x) = grad1.(grad1(x))

###############################################################################
# end of cloud computing
###############################################################################
function disablecache()
    pkgs = readdir(Pkg.dir())
    pkgs = symdiff(pkgs, ["METADATA", "META_BRANCH", "REQUIRE", ".cache", ".trash"])
    for pkg in pkgs
        modulefile = joinpath(Pkg.dir(pkg), "src", pkg * ".jl")
        code = readstring(modulefile)
        code = replace(code, "__precompile__()", "__precompile__(false)")
        code = replace(code, "__precompile__(true)", "__precompile__(false)")
        write(modulefile, code)
    end
end

hostnames(pool = workers())  = pmap(WorkerPool(pool), x -> readlines(`hostname`), 1:length(pool))

export @headnode
macro headnode(ex)
    quote
        @everywhere using Utils
        nodes = hostnames()
        @sync @parallel for i in 1:length(workers())
            if i == findfirst(x -> x == nodes[i], nodes)
                $(esc(ex))
            end
        end
    end
end

export aws_setup
function aws_setup(n = 0)
    Sys.CPU_CORES > 2 && return
    @eval using ClusterManagers; addprocs_qrsh(n)
    @headnode begin
        try run(`sudo mkfs -t ext4  /dev/nvme0n1`) end
        try run(`sudo mkdir /scratch`) end
        try run(`sudo mount /dev/nvme0n1 /scratch`) end
        run(`awk 'BEGIN {cmd="sudo cp -ri /shared/Data /scratch/"; print "n" |cmd;}'`)
        run(`sudo chmod -R ugo+rw /scratch`)
    end
end

export scc_setup
function scc_setup()
    if Sys.CPU_CORES > 8 && !contains(readstring(`hostname`), "highchain")
        @eval begin
            using MPI
            MPI.Init()
            mngr = MPI.start_main_loop(MPI.MPI_TRANSPORT_ALL)
        end
    end
end

export scc_end
function scc_end()
    isdefined(Main, :MPI) && @everywhere (MPI.Finalize(); exit())
    exit()
end


function getjobid(cmd)
    println(string(cmd)[2:end-1])
    jobstr = readstring(cmd)
    jobid = match(r"<(\d+)>", jobstr).captures[1]
end

bkill(ids) = for id in ids try run(`bkill $id`) end end

export bsub
function bsub(;queue = "smallopa", subqueue = queue, n = 28, fn = "main")
    jobid = getjobid(`bsub -q $queue -n $n -oo $fn.log -eo $fn.err echo`)
    local nodes
    while true
        jobs = readstring(`bjobs $jobid`)
        nodes = [m.match for m in eachmatch(r"node(\d+)", jobs)]
        nodes = setdiff(nodes, [readchomp(`hostname`)])
        !isempty(nodes) && break
        sleep(1)
    end
    write("$(fn)_compile.jl", readline("$fn.jl"), "using MPI; exit()")
    jobids = [getjobid(`bsub -q $subqueue -m $node -eo $node.err julia $(fn)_compile.jl`) for node in nodes]
    w = join(["done($jobid)" for jobid in jobids], "&&")
    nodes = join(nodes, " ")
    run(`bsub -q $queue -n $n -w $w -m $nodes -oo $fn.log -eo $fn.err -J $fn mpijob-new julia $fn.jl`)
end

###############################################################################
# end of cloud computing
###############################################################################

###############################################################################
# begin of TypedTables
###############################################################################

import TypedTables: @Table, @Row

macro Table(exprs...)
    N = length(exprs)
    names = Vector{Any}(N)
    values = Vector{Any}(N)
    for i = 1:N
        expr = exprs[i]
		if isa(expr, Symbol)
			names[i] = expr
			values[i] = esc(expr)
        elseif isa(expr.args[1],Symbol)
            names[i] = (expr.args[1])
            values[i] = esc(expr.args[2])
        elseif isa(expr.args[1],Expr)
            if expr.args[1].head != :(::) || length(expr.args[1].args) != 2
                error("A Expecting expression like @Table(name1::Type1 = value1, name2::Type2 = value2) or @Table(name1 = value1, name2 = value2)")
            end
            names[i] = (expr.args[1].args[1])
            values[i] = esc(Expr(:call, :convert, expr.args[1].args[2], expr.args[2]))
        else
            error("A Expecting expression like @Table(name1::Type1 = value1, name2::Type2 = value2) or @Table(name1 = value1, name2 = value2)")
        end
    end
    tabletype = TypedTables.Table{(names...)}
    return Expr(:call, tabletype, Expr(:tuple, values...))
end

macro Row(exprs...)
    N = length(exprs)
    names = Vector{Any}(N)
    values = Vector{Any}(N)
    for i = 1:N
        expr = exprs[i]
		if isa(expr, Symbol)
			names[i] = expr
			values[i] = esc(expr)
        elseif isa(expr.args[1],Symbol)
            names[i] = (expr.args[1])
            values[i] = esc(expr.args[2])
        elseif isa(expr.args[1],Expr)
            if expr.args[1].head != :(::) || length(expr.args[1].args) != 2
                error("A Expecting expression like @Row(name1::Type1 = value1, name2::Type2 = value2) or @Row(name1 = value1, name2 = value2)")
            end
            names[i] = (expr.args[1].args[1])
            values[i] = esc(Expr(:call, :convert, expr.args[1].args[2], expr.args[2]))
        else
            error("A Expecting expression like @Row(name1::Type1 = value1, name2::Type2 = value2) or @Row(name1 = value1, name2 = value2)")
        end
    end
    rowtype = TypedTables.Row{(names...)}
    return Expr(:call, rowtype, Expr(:tuple, values...))
end

DataFrames.DataFrame{Names, StorageTypes}(tbl::TypedTables.Table{Names, StorageTypes}) = DataFrame(collect(tbl.data), collect(Names))

TypedTables.Table(df::DataFrame) = Table(DataFrames.columns(df), names(df))

function TypedTables.Table(column_eltypes::Vector{DataType}, names::Vector{Symbol})
	column_types =Tuple{[Vector{typ} for typ in column_eltypes]...}
	Table{tuple(names...), column_types}()
end

function TypedTables.Table(columns::Vector, names::Vector{Symbol})
	Table{tuple(names...)}(tuple(columns...))
end

function Base.push!(tbl::TypedTables.Table, row)
	for (col, val) in zip(tbl.data, row)
		push!(col, val)
	end
end

Base.Matrix(tbl::TypedTables.Table) = hcat(tbl.data...)

Base.writedlm(f::AbstractString, tbl::TypedTables.Table, delim = ','; opts...) = writedlm(f, Matrix(tbl), delim; opts...)

###############################################################################
# end of TypedTables
###############################################################################

###############################################################################
# begin of macros
###############################################################################

export @nt
macro nt(exs...)
  esc(:(@NT($(exs...))($(exs...))))
end

export @unstruct
macro unstruct(typ)
    blk = Expr(:block)
    for f in fieldnames(eval(current_module(), typ))
        push!(blk.args, :($f = getfield($typ, $(QuoteNode(f)))))
    end
    esc(blk)
end

macro unstruct(typ, exs...)
  blk = Expr(:block)
  for ex in exs
    exquot = QuoteNode(ex)
    push!(blk.args, :($ex =getfield($typ, $exquot)))
  end
  esc(blk)
end

export @symdict
"""
    a = 1; b = 2
    d = @symdict(a, b)
"""
macro symdict(exs...)
  expr = Expr(:block,:(d = Dict()))
  for ex in exs
    push!(expr.args,:(d[$(QuoteNode(ex))] = $(esc(ex))))
  end
  push!(expr.args,:(d))
  expr
end

export @strdict
"""
    a = 1; b = 2
    d = @strdict(a, b)
"""
macro strdict(exs...)
  expr = Expr(:block,:(d = Dict()))
  for ex in exs
    push!(expr.args,:(d[$(string(ex))] = $(esc(ex))))
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

export @undict
"""
    d = Dict("a"=>[1,2,3], "b" => 2)
    @undict d a b
    d = Dict(:a=>[1,2,3], :b => 2)
    @undict d a b
"""
macro undict(d, exs...)
  blk = Expr(:block)
  for ex in exs
    exquot = QuoteNode(ex)
    exstr = string(ex)
    push!(blk.args, :($ex = haskey($d, $exquot) ? $d[$exquot] : $d[$exstr]))
  end
  esc(blk)
end

###############################################################################
# end of macros
###############################################################################

macro logto(fn)
  quote
    logstream = open(joinpath(tempdir(), $fn * ".log"), "w")
    Logging.configure(output = [logstream, STDOUT], level = Logging.DEBUG)
  end |> esc
end

export @raw_str
macro raw_str(s) s end

export mat2img, img2mat, vec2rgb

function mat2img(A)
  @eval using Images
  colorview(RGB, permutedims(A, (3, 1, 2)))
end

function img2mat(A)
  @eval using Images
  @> A channelview permutedims((2, 3, 1))
end

vec2rgb(x) = (W = Int(√(length(x) ÷ 3)); reshape(x, W, W, 3))

export pmapreduce
pmapreduce(f, op, iter) = reduce(op, pmap(f, iter))

Base.vec(x::Number) = fill(x, 1)
Base.vec(x::Symbol) = fill(x, 1)
Base.vec(x::Expr) = x.args

export @pygen

macro pygen(f)
    # generate a new symbol for the channel name and
    # wrapper function
    c = gensym()
    η = gensym()
    # yield(λ) → put!(c, λ)
    f′ = MacroTools.postwalk(f) do x
        @capture(x, yield(λ_)) || return x
        return :(put!($c, $λ))
    end
    # Fetch the function name and args
    @capture(f′, function func_(args__) body_ end)
    # wrap up the η function
    final = quote
        function $func($(args...))
            function $η($c)
                $body
            end
            return Channel(c -> $η(c))
        end
    end
    return esc(final)
end


function Base.edit(path::AbstractString, line::Integer=0)
    command = Base.editor()
    name = basename(first(command))
    issrc = length(path)>2 && path[end-2:end] == ".jl"
    if issrc
        f = Base.find_source_file(path)
        f !== nothing && (path = f)
    end
    cmd = line != 0 ? `$command -g $path:$line` : `$command $path`
    spawn(pipeline(cmd, stderr=STDERR))
    nothing
end

Base.get(dict::Dict, key) = Base.get(dict, key, nothing)

export setdefault
"""
    d = Dict(:a => 1)
    get(d, :a)
    setdefault!(d, :a, 2)
    setdefault!(d, :b, 2)
"""
setdefault!(dict::Dict, key, value) = haskey(dict, key) ? dict[key] : (dict[key] = value)


function area(coords)
  b = coords[end]
  v = 0.0
  for i in 1:length(coords)
    a, b = b, coords[i]
    v += a[2] * b[1] - a[1] * b[2]
  end
  return v * 0.5
end


export confusmat
function confusmat(ul, y, ypred)
  encoder = LabelEncoder(ul)
  ypred_int = transform(encoder, ypred) + 1
  y_int = transform(encoder, y) + 1
  R = Int[countnz((y_int .== i) .& (ypred_int .== j)) for i in 1:len(ul), j in 1:len(ul)]
  mat = Any["gt/pred" ul'; Any[ul R]]
end

export mat2acc
mat2acc(mat) = sum(diag(mat)) / sum(mat)


###############################################################################
# type try
###############################################################################
export @ignore
macro ignore(ex)
  :(try $ex; catch e; warn(e); end) |> esc
end

export @catch
macro catch(ex)
  :(try $ex; catch e; warn(e); return; end) |> esc
end

export @correct
macro correct(ex)
  res = gensym()
  :($res = try $ex end; $res == nothing ? false : $res) |> esc
end

export ntry
macro ntry(ex, n = 1000)
	:(for t in 1:$n
  		try $ex; break; catch e; warn(e);	end
  	end) |> esc
end

export @trys
macro trys(exs...)
  expr = :()
  for ex in exs[end:-1:1]
    expr = :(try $ex; catch e; $expr; end)
  end
  esc(expr)
end
###############################################################################
# type try
###############################################################################

###############################################################################
# type traits
###############################################################################
export @abstrait, @trait, @mixin

traits_declarations = Dict{Symbol, Array}()

macro abstrait(typedef)
  typedef = macroexpand(typedef)
  declare, block = typedef.args[2:3]
  sym = (@correct declare.head == :<:) ? declare.args[1] : declare
  traits_declarations[sym] = block.args
  Expr(:abstract, declare) |> esc
end

macro trait(typedef)
  typedef = macroexpand(typedef)
  declare, block = typedef.args[2:3]
  sym = (@correct declare.head == :<:) ? declare.args[1] : declare
  traits_declarations[sym] = block.args
  esc(typedef)
end

macro mixin(typedef)
  typedef = macroexpand(typedef)
  head = typedef.head
  sym, parents = typedef.args[2].args
  parents = vec(parents)
  block = typedef.args[3]
  for parent in parents
    parentfields = get(traits_declarations, parent, Expr[])
    append!(block.args, parentfields)
  end
  field = OrderedDict()
  for arg in block.args
    @capture(arg, (f_::typ_=val_)|(f_::typ_)|(f_=val_))
    f == nothing && (f = deepcopy(arg))
    get!(field, f, deepcopy(arg))
  end
  block.args = collect(values(field))
  ind = findfirst(x -> isempty(fieldnames(eval(current_module(), x))), parents)
  if ind == 0
    return Expr(head, true, sym, block) |> esc
  else
    return Expr(head, true, Expr(:<:, sym, parents[ind]), block) |> esc
  end
end

# macro inherit(typedef)
#   typedef = macroexpand(typedef)
#   head = typedef.head
#   sym, parent = typedef.args[2].args
#   block = typedef.args[3]
#   prepend!(block.args, get(traits_declarations, parent, Expr[]))
#   Expr(head, true, Expr(:<:, sym, parent), block) |> esc
# end
###############################################################################
# end of type traits
###############################################################################

export capitalize
capitalize = uppercase

Base.find(s::String, c::Union{Char, Vector{Char}, String, Regex}, start = 1) = search(s, c, start)

Base.delete!(a::Array, val) = deleteat!(a, findfirst(a, val))

export deleteall!, remove!, removeall!
deleteall!(a::Array, val)  = deleteat!(a, find(x -> x==val, a))
remove!(a::Array, val) = delete!(a, val)
removeall!(a::Array, val) = deleteall!(a, val)

export hasfield
hasfield(x, s) = isdefined(x, s)

function paper(fn)
	download("https://raw.githubusercontent.com/ihrke/markdown-paper/master/templates/elsarticle-template-1-num.latex", "elsarticle-template-1-num.latex")
	run(`pandoc $fn.md
			-s -S -o $fn.pdf
			--filter=pandoc-crossref
			--filter=pandoc-citeproc
			--template=elsarticle-template-1-num.latex
			--bibliography=references.bib`)
	rm("elsarticle-template-1-num.latex")
end


const STYLEPATH = "https://raw.githubusercontent.com/tompollard/phd_thesis_markdown/master/style"
# const STYLEPATH = "https://raw.githubusercontent.com/AStupidBear/phd_thesis_markdown/master/style"

function thesis(fn, fmt = "pdf"; title = "This is the title of the thesis", name = "Yao Lu")
  if fmt == "pdf"
    download("$STYLEPATH/template.tex", "template.tex")
    download("$STYLEPATH/preamble.tex", "preamble.tex")
    run(`pandoc $(glob("*.md"))
        -o $fn.pdf
        --filter=pandoc-crossref
        --filter=pandoc-citeproc
        --include-in-header=preamble.tex
        --template=template.tex
        --bibliography=references.bib
        --csl=$STYLEPATH/ref_format.csl
        --highlight-style=pygments
        --variable=fontsize:12pt
        --variable=papersize:a4paper
        --variable=documentclass:report
        --number-sections
        --latex-engine=xelatex`)
    rm("template.tex")
    rm("preamble.tex")
  elseif fmt == "html"
    download("$STYLEPATH/template.html", "template.html")
    download("$STYLEPATH/style.css", "style.css")
    @>(readstring("template.html"),
    replace("This is the title of the thesis", title),
    replace("Firstname Surname", name)) |>
    x -> write("template.html", x)
    run(`pandoc $(glob("*.md"))
        -o $fn.html
        --standalone
        --filter=pandoc-crossref
        --filter=pandoc-citeproc
        --template=template.html
        --bibliography=references.bib
        --csl=$STYLEPATH/ref_format.csl
        --include-in-header=style.css
        --toc
        --number-sections
        --mathjax`)
    rm("template.html")
    rm("style.css")
  end
end


"""
    file = "A unified approach to building and controlling spiking attractor networks"
    name2bib(file)
    clipboard(join([file, "12"],"\n"))
    name2bib()
"""
function name2bib(file::AbstractString; issn="", volume="", issue="", pages="")
  try
    filep = replace(file, " ", "+")
    url = "http://libgen.io/scimag/index.php?s=$filep&journalid=$issn&&v=$volume&i=$issue&p=$pages&redirect=1"
    content = url |> download |> readstring

    pat = r"<a href=\"(.*)\"  title=.*>Libgen"
    url = match(pat, content).captures[1] |> string
    content = url |> download |> readstring

    pat = r"<textarea.*>(.*)</textarea>"s
    bib = match(pat, content).captures[1] |> string
  catch
    file
  end
end

function name2bib()
  files = strip.(split(clipboard(),"\n"))
  filter!(x->!isempty(x), files)

  fails = []; succs = []
  for file in files
    bib = bibtex(file)
    if bib != file
      push!(succs, bib)
    else
      push!(fails, file)
    end
  end

  bibs = join([succs; fails],"\n")
  println(bibs)
  clipboard(bibs)
  bibs
end


"""
    file = "A unified approach to building and controlling spiking attractor networks"
    libgen(file)
    clipboard(join([file,"12"],"\n"))
    libgen()
"""
function libgen(file::AbstractString; issn="", volume="", issue="", pages="")
  try
    filep = replace(file, " ", "+")
    url = "http://libgen.io/scimag/index.php?s=$filep&journalid=$issn&&v=$volume&i=$issue&p=$pages&redirect=1"
    content = url |> download |> readstring

    pat = r"<a href=\"(.*)\"  title=.*>Libgen"
    url = match(pat, content).captures[1] |> string
    content = url |> download |> readstring

    pat = r"<a href='(.*)'><h2>DOWNLOAD</h2>"
    url = match(pat, content).captures[1] |> string
  catch
    file
  end
end

function libgen()
  files = strip.(split(clipboard(), "\n"))
  filter!(x->!isempty(x), files)

  fails = []; succs = []
  for file in files
    url = libgen(file)
    if url != file
      push!(succs, url)
    else
      push!(fails, file)
    end
  end

  urls = join([succs; fails],"\n")
  println(urls)
  clipboard(urls)
  urls
end

"""
doi2cit("10.1126/science.169.3946.635")
"""
doi2cit(doi) = readstring(`curl -LH "Accept: text/x-bibliography; style=apa" https://doi.org/$doi -k`)
"""
doi2bib("10.1126/science.169.3946.635")
"""
doi2bib(doi) = readstring(`curl -LH "Accept: application/x-bibtex" https://doi.org/$doi -k`)

"""
markdown-preview-enhanced

# Examples

```{julia output:"html", id:"hehe"}
using Plots; plot(rand(10)) |> mpe
```
"""
function mpe(p, fmt::Symbol = :svg)
    p.attr[:html_output_format] = fmt
    show(STDOUT, MIME("text/html"), p)
end

export parseweb
function parseweb(url; relative = false, parent = false)
  opts = `--continue --recursive --convert-links --html-extension
              --page-requisites --no-check-certificate`
  relative == true && (opts = `$opts --relative`)
  parent == false && (opts = `$opts --no-parent`)
  run(`wget $opts $url`)
end

export viewdf
function viewdf(df)
  fn = tempname() * ".csv"
  writetable(fn, df)
  is_windows() && spawn(`csvfileview $fn`)
end

export viewmat
function viewmat(x)
  fn = tempname() * ".csv"
  writecsv(fn, vcat(rowvec(1:size(x, 2)), x))
  is_windows() && spawn(`csvfileview $fn`)
end

export sign
Base.sign(x::Real, Θ) = ifelse(x < -Θ, oftype(x, -1), ifelse(x > Θ, one(x), zero(x)))

function rename_youtube()
  for (root, dirs, files) in walkdir(pwd())
    for file in files
      fn = joinpath(root, file)
      fn_new = @>(fn, replace(r" \(.*\)", ""),
                  replace("%3", " "), replace("_hd", ""))
      try mv(fn, fn_new) end
    end
  end
end

export undersample
function undersample(xs::Array...; nsample = 2, featuredim = "col")
	if featuredim == "col"
		getfun, setfun, countfun = cview, cset!, ccount
	elseif featuredim == "row"
		getfun, setfun, countfun = rview, rset!, rcount
	end
  cols = rand(1:countfun(xs[1]), nsample)
  [getfun(x, cols) for x in xs]
end

export plus, minus
plus(x::Real) = ifelse(x > 0, one(x), zero(x))
minus(x::Real) = ifelse(x < 0, oftype(x, -1), zero(x))

export splat
# splat(x) = @as _ x collect.(_) vec.(_) vcat(_...)
splat(list) = [item for sublist in list for item in sublist]

export git, jgit
jgit() = git(pwd(), ".jl")
function git(path = pwd(), suffix = "")
	folder = splitdir(path)[end]
	cd(path)
  run(`git config --global user.name "Yao Lu"`)
  run(`git config --global user.email "luyaocns@gmail.com"`)
	run(`git init`)
	try run(`git remote add $folder git@github.com:AStupidBear/$folder$suffix.git`) end
	try run(`git pull $folder master`) end
	run(`git add .`)
	try run(`git commit -m $(now())`) end
	run(`git push $folder master`)
end

export typename
function typename{T}(x::T)
  name, ext = splitext(string(T.name))
  isempty(ext) ? name : ext[2:end]
end

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

function Base.shuffle(x::Union{AbstractArray, Tuple}, y::Union{AbstractArray, Tuple}; featuredim = "col")
	if featuredim == "col"
		getfun, setfun, countfun = cview, cset!, ccount
	elseif featuredim == "row"
		getfun, setfun, countfun = rview, rset!, rcount
	end
  a = randperm(countfun(y))
  x, y = getfun(x, a), getfun(y, a)
end

export typeparam
typeparam{T}(x::T) = T.parameters[1]

export memory
memory(x) = Base.summarysize(x) / 1024^2

"""cron("spam.jl", 1)"""
function cron(fn, repeat)
  name = splitext(fn)[1]
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

macro debug(ex)
  x = gensym()
  :($x = $ex; debug($x); $x) |> esc
end
macro info(ex)
  x = gensym()
  :($x = $ex; info($x); $x) |> esc
end

function args2field(args)
    fields = Symbol[]
    for arg in args
      @capture(arg, (f_::typ_=val_)|(f_::typ_)|(f_=val_))
      f != nothing && push!(fields, f)
    end
    return fields
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
    for (typ, n) in zip(typs, names)
        field = fieldnames(eval(current_module(), typ))
        field = union(field, args2field(get(traits_declarations, typ, Symbol[])))
        for f in field
            exreplace!(ex.args[2], :($f), :($n.$f))
        end
    end
    # @show ex
    esc(ex)
end

# """
# 	X = [([1,2,3],[4,5,6]), ([1,2,3], [4,5,6])]
# 	vcat(X) == vcat(X...)
# """
for f in (:vcat, :hcat)
  @eval begin
    function Base.$f(X::Tuple...)
      ntuple(length(X[1])) do j
        $f([X[i][j] for i in 1:length(X)]...)
        # mapreduce(i -> X[i][j], $f, 1:length(X))
      end
    end
  end
end
###############################################################################
# end of cget
###############################################################################
export csize, clength, ccount, cview, cget, cset!, size2
csize(a) = (ndims(a) == 1 ? size(a) : size(a)[1:end-1])
csize(a, n) = tuple(csize(a)..., n) # size if you had n columns
clength(a) = (ndims(a) == 1 ? 1 : stride(a, ndims(a)))
ccount(a) = (ndims(a) == 1 ? length(a) : size(a, ndims(a)))
cview(a, i) = (ndims(a) == 1 ? a[i] : view(a, ntuple(i->(:), ndims(a) - 1)..., i))
cget(a, i) = (ndims(a) == 1 ? a[i] : getindex(a, ntuple(i->(:), ndims(a)-1)..., i))
cset!(a, x, i) = (ndims(a) == 1 ? (a[i] = x) : setindex!(a, x, ntuple(i->(:), ndims(a) - 1)..., i))
size2(y) = (nd = ndims(y); (nd == 1 ? (length(y), 1) : (stride(y, nd), size(y, nd)))) # size as a matrix
size2(y, i) = size2(y)[i]

export rsize, rlength, rcount, rview, rget, rset!, size1
rsize(a) = (ndims(a)==1 ? size(a) : size(a)[2:end])
rsize(a, n) = tuple(n, rsize(a)...) # size if you had n columns
rlength(a) = (ndims(a) == 1 ? length(a) : stride(a, ndims(a)))
rcount(a) = (ndims(a) == 1 ? length(a) : size(a, 1))
rview(a, i) = (ndims(a) == 1 ? a[i] : view(a, i, ntuple(i->(:), ndims(a) - 1)...))
rget(a, i) = (ndims(a) == 1 ? a[i] : getindex(a, i, ntuple(i->(:), ndims(a)-1)...))
rset!(a, x, i) = (ndims(a)==1 ? (a[i] = x) : setindex!(a, x, i, ntuple(i->(:), ndims(a)-1)...))
size1(y) = (nd = ndims(y); (nd == 1 ? (length(y), 1) : (size(y, 1), prod(size(y)[2:end])))) # size as a matrix
size1(y, i) = size1(y)[i]

for s in (:cget, :rget, :cview, :rview)
  @eval $s(as::Tuple, i) = tuple([$s(a, i) for a in as]...)
end
for s in (:cset!, :rset!)
  @eval $s(as::Tuple, xs::Tuple, i) = for (a, x) in zip(as, xs); $s(a, x, i); end
end

ccount(x::Tuple) = ccount(x[1])
###############################################################################
# end of cget
###############################################################################
Base.similar(x::Tuple) = deepcopy(x)
export balance
"""
    using Utils
    x = rand(4, 100)
    y = randprob(-1:1, [0.2, 0.6, 0.2], 100)
    xb, yb = balance(x, y)
    hist1(y, -1.5:1.5)
    hist1(yb, -1.5:1.5)
"""
function balance(x, y; featuredim = "col", sampleratio = 1.0)
	if featuredim == "col"
		getfun, setfun, countfun = cview, cset!, ccount
	elseif featuredim == "row"
		getfun, setfun, countfun = rview, rset!, rcount
	end

  d = Dict()
  for i in 1:ccount(x)
    get!(d, y[i], [getfun(x, i)])
    push!(d[y[i]], getfun(x, i))
  end

  nb = Int(countfun(x) * sampleratio); yb = zeros(nb)
  xb = featuredim == "col" ? zeros(size(x, 1), nb) : zeros(nb, size(x, 2))

  ny, key, vals = length(d), collect(keys(d)), collect(values(d))
  for i in 1:nb
    r = rand(1:ny)
    setfun(yb, key[r], i)
    setfun(xb, rand(vals[r]), i)
  end
  return xb, yb
end


export readabsdir
readabsdir(dir) = map(file->joinpath(dir, file), readdir(dir))

export hasnan
function hasnan(x)
  for i in eachindex(x)
    isnan(x[i]) && return true
  end
  false
end

function readall(f::IO, T)
  x = Vector{T}()
  while !eof(f)
    push!(x, read(f, T))
  end
  x
end

function readall(fn::AbstractString, T)
  x = Vector{T}()
  open(fn, "r") do f
    while !eof(f); push!(x, read(f, T));  end
  end
  x
end

export centralize, centralize!
"transform x to (-1, 1)"
function centralize!(x, dim=1)
  𝑚ax = maximum(x, dim)
  𝑚in = minimum(x, dim)
  x .= (x .- 𝑚in) ./ (𝑚ax .- 𝑚in)
  x .= 2 .* x .- 1
end
centralize(x, dim = 1) = centralize!(deepcopy(x), dim)

export minute
minute() = @> string(now())[1:16] replace(":", "-")

export date
date() = string(now())[1:10]

export timename
timename(fn) = joinpath(tempdir(), minute() * "_" * fn)


export histn
"c, w = histn(rand(10), rand(10), rand(10))"
function histn(xs::Array...; o...)
	h = StatsBase.fit(StatsBase.Histogram, vec.(xs); o...)
	edge2center.(h.edges), h.weights
end

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
abstract type Parameter end

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

export @withkw
macro withkw(ex)
  ex = macroexpand(ex)
  :(Parameters.@with_kw $ex) |> esc
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
      tmp = arg.args[2]
      if VERSION <= v"0.5.2"
        if isa(tmp, Expr) && tmp.head == :(=>)
          push!(bounds, tmp.args[2])
          arg.args[2] = tmp.args[1]
        else
          push!(bounds, tmp)
        end
      else
        if isa(tmp, Expr) && tmp.args[1] == :(=>)
          push!(bounds, tmp.args[3])
          arg.args[2] = tmp.args[2]
        else
          push!(bounds, tmp)
        end
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
      tmp = deepcopy(args[i].args)
      deleteat!(args, i)
      insert!(args, i, tmp)
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
unroll(x, name) = [getfield(x, s) for s in fieldnames(x) if contains(string(s), string(name))]

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

ustc() = transfer_system("172.16.1.17", "luyao", 22)

highchain() = transfer_system("101.231.45.146", "luyao", 8822)

export linux_path, cygdrive

linux_path(path) = replace(path, "\\", "/")

cygdrive(path) = @> path linux_path replace(":", "") x->"/cygdrive/$x"

function junocloud(ip, user, port)
  local_root = joinpath(homedir(), "Documents", "Codes") |> linux_path
  remote_root = "/home/$user/Documents"

  m = match(r"connect\((.*)\)", clipboard())
  remoteport = is(m, nothing) ? "55440" : m[1]
  localport = "1234"

  julia_eval = """using Juno;Juno.connect(1234)"""
  ssh_eval = """chmod 400 ~/.ssh/id_rsa; ~/julia-0.5/bin/julia -i -e "$julia_eval"; bash"""

  src= joinpath(homedir(), ".juliarc.jl")
  dst = "$user@$ip:/home/$user/.juliarc.jl"
  rsync(src, dst, port)

  src = joinpath(homedir(), ".ssh", "id_rsa")
  dst = "$user@$ip:/home/$user/.ssh/id_rsa"
  rsync(src, dst, port)

  src = local_root
  dst = "$user@$ip:$remote_root"
  rsync(src, dst, port)

  cmd = `ssh -X -R $localport:localhost:$remoteport -p $port
     $user@$ip -t $ssh_eval`
  run(cmd)
end
###############################################################################
# end of Cloud
###############################################################################

export colvec, rowvec
colvec(x) = reshape(x, length(x), 1)
rowvec(x) = reshape(x, 1, length(x))

export labelplot
function labelplot(x, label)
  p = Main.plot(; legend = nothing)
  t0 = 1
  for t in 2:length(x)
    if label[t] != label[t-1] || t == length(x)
      Main.plot!(p, t0:t, x[t0:t]; color = Int(label[t-1]), label = label[t-1])
      t0 = t
    end
  end
  return p
end

function imconvert(ext1, ext2)
  for (root, dirs, files) in walkdir(pwd())
    for file in files
      name, ext = splitext(joinpath(root, file))
      if ext == ext1
        name1 = name * ext1
        name2 = name * ext2
        run(`imconvert $name1 $name2`)
      end
    end
  end
end

export fieldvalues, fields
fieldvalues(x) = [getfield(x, s) for s in fieldnames(x)]
fields(x) = [(s, getfield(x, s)) for s in fieldnames(x)]

###############################################################################
# LabelEncoder
###############################################################################
export LabelEncoder, fit_transform, fit, transform, inverse_transform
@with_kw type LabelEncoder
  unique_label::Array = []
end

Base.length(encoder::LabelEncoder) = length(encoder.unique_label)

function fit_transform(encoder::LabelEncoder, label)
  fit(encoder, label)
  transform(encoder, label)
end

function fit(encoder::LabelEncoder, label)
  encoder.unique_label = unique(label)
  sort!(encoder.unique_label)
end

function transform(encoder::LabelEncoder, label)
  [findfirst(encoder.unique_label, l) - 1 for l in label]
end

function inverse_transform(encoder::LabelEncoder, index)
  [encoder.unique_label[Int(i + 1)] for i in index]
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
  sort!(encoder.unique_label)
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
  𝑚ax::Array{Float32} = []
  𝑚in::Array{Float32} = []
end

function fit_transform(scaler::MinMaxScaler, x, shape = (); dim = 2, reshape = false)
  fit(scaler, x; dim = dim)
  transform(scaler, x)
end

function fit(scaler::MinMaxScaler, x, shape = (); dim = 2, reshape = false)
  scaler.sample_dim, scaler.𝑚ax, scaler.𝑚in = dim, maximum(x, 2), minimum(x, 2)
  return scaler
end

transform(scaler::MinMaxScaler, x, shape = (); reshape = false) = (Array{Float32}(x) .- scaler.𝑚in) ./ (scaler.𝑚ax .- scaler.𝑚in .+ 1.0f-20) .- 0.5f0

inverse_transform(scaler::MinMaxScaler, x, shape = (); reshape = false) = (Array{Float32}(x) .+ 0.5f0) .* (scaler.𝑚ax .- scaler.𝑚in + 1.0f-20) .+ scaler.𝑚in

###############################################################################
# end of MinMaxScaler
###############################################################################

###############################################################################
# ShapeScaler
###############################################################################
export ShapeScaler, fit_transform, fit, transform, inverse_transform

"""
    scaler = ShapeScaler()
    x = ([1 2 3 4; -1 -2 -3 -4; 1 2 3 4; 2 2 3 4], )
    fit_transform(scaler, x, ((2, 2),))
    fit_transform(scaler, x, ((2, 2),); reshape = false)
    fit(scaler, x, ((2, 2),))
    transform(scaler, x)
"""

@with_kw type ShapeScaler
  scalers::Vector{MinMaxScaler} = Vector{MinMaxScaler}()
  shapes::Tuple = ()
end

function fit_transform(scaler::ShapeScaler, x, shapes; reshape = false)
  fit(scaler, x, shapes)
  transform(scaler, x; reshape = reshape)
end

function fit(scaler::ShapeScaler, x, shapes)
  scaler.shapes = shapes
  for xi in x  push!(scaler.scalers, fit(MinMaxScaler(), xi)) end
end

function transform(scaler::ShapeScaler, x; reshape = false)
  ntuple(length(x)) do i
    xi = transform(scaler.scalers[i], x[i])
    reshape ? Base.reshape(xi, (scaler.shapes[i]..., ccount(xi))) : xi
  end
end

###############################################################################
# end of ShapeScaler
###############################################################################

###############################################################################
# ImageScaler
###############################################################################
export ImageScaler, fit_transform, fit, transform, inverse_transform

"""
    scaler = ImageScaler()
    x = [1 2 3 4; -1 -2 -3 -4; 1 2 3 4; 2 2 3 4]
    fit_transform(scaler, x, (2, 2))
    fit_transform(scaler, x, (2, 2); reshape = false)
    fit(scaler, x, (2, 2))
    transform(scaler, x)
"""
@with_kw type ImageScaler
  𝑚ax::Float32 = 1
  shape::NTuple = ()
end

function fit_transform(scaler::ImageScaler, x, shape; reshape = false)
  fit(scaler, x, shape)
  transform(scaler, x; reshape = reshape)
end

fit(scaler::ImageScaler, x, shape) = (scaler.shape = shape; scaler.𝑚ax = maximum(abs, x))

function transform(scaler::ImageScaler, x; reshape = false)
  xs = Array{Float32}(x) / scaler.𝑚ax
  reshape ? Base.reshape(xs, (scaler.shape..., ccount(xs))) : xs
end
###############################################################################
# end of ImageScaler
###############################################################################

export softmax, hardmax
softmax(x, dim = 1) = (y = exp.(x); y ./ sum(y, dim))
hardmax(x, dim = 1) = x .== maximum(x, dim)

macro curry(n, f)
    syms = [gensym() for i=1:n]
    foldl((ex, sym)  -> Expr(:->, sym, ex), Expr(:call, f, syms...), reverse(syms))
end

export partial
partial(f, a...; b...) = ((x...; y...) -> f(a..., x...; b..., y...))


export @print
macro print(expr...)
  str = join(["$ex = %.2f " for ex in expr], ", ")
  esc(:(@printf($str, $(expr...))))
end

Base.max(itr) = maximum(itr)
Base.min(itr) = minimum(itr)

###############################################################################
# Information Theory
###############################################################################
export xcov
xcov(A,B) = xcorr(A .- mean(A), B .- mean(B))

export entropy
"entropy(randn(10000))"
function entropy(x)
	ε = 1e-100
	c, w = histn(x)
	P = w / sum(w)
	H = sum(-P .* log.(2, P .+ ε))
	scale = log(2, prod(length.(c)))
	return H / scale
end

export mutualinfo
"mutualinfo(randn(10000), randn(10000))"
function mutualinfo(x, y)
	ε = 1e-100
	c, w = histn(x, y)
	Pxy = w / sum(w)
	Px = sum(Pxy, 2)
	Py = sum(Pxy, 1)
	Hx = sum(-Px .* log.(2, Px .+ ε))
	Hy = sum(-Py .* log.(2, Py .+ ε))
	I = sum(Pxy .* log.(2, Pxy ./ (Px .* Py .+ ε) .+ ε))
	return I / Hy
end

export kl
"kl(randn(100000), randn(100000) + 10)"
function kl(x1, x2)
	ε = 1e-100
	low, up = extrema((extrema(x1)..., extrema(x2)...))
	edge = linspace(low, up, 1000)
	P1 = @as _ x1 fit(Histogram, vec(_), edge) _ ./ sum(_)
	P2 = @as _ x2 fit(Histogram, vec(_), edge) _ ./ sum(_)
	KL = sum(P1 .* log.(2, P1 ./ (P2 .+ ε) .+ ε))
end
###############################################################################
# end of Information Theory
###############################################################################

###############################################################################
# Distributions
###############################################################################
export edge2center
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
###############################################################################
# end of Distributions
###############################################################################


"""
    using Distributions
    rand(MvNormal([1.,2.,3.], [1.,1.,1.]), 1000)
"""
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

export cutoff
heaviside(x) = 0.5 * (1 + sign(x))
delta(x, δx = 1e-3) = (heaviside(x + δx / 2) - heaviside(x - δx / 2)) / δx
interval(x, xmin, xmax) = x > xmax ? 0.0 : (x < xmin ? 0.0 : 1.0)
cutoff{T}(x::T, xmin, xmax)::T =   x < xmin ? xmin :
                                   x > xmax ? xmax : x
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

export forwdiffn
function forwdiffn(n, x, y = x)
  dx = zeros(x)
  for t in 1:(length(x) - n)
    dx[t] = x[t + n] - y[t]
  end
  dx
end

export diffn
function diffn(n, x, y = x)
  dx = zeros(x)
  for t in (n + 1):length(x)
    dx[t] = x[t] - y[t-n]
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
  Expr(:block, (ex for i in 1:n)...) |> esc
end

"""
    aria2c("https://www.youtube.com")
    aria2c("https://www.youtube.com"; proxy = "127.0.0.1:1080")
"""
function aria2c(url, fn; proxy = "")
  dir, base  = splitdir(fn)
  run(`aria2c --max-connection-per-server=8 --all-proxy=$proxy -d $dir -o $base $url`)
  fn
end
aria2c(url; o...) = aria2c(url, tempname(); o...)

export psdownload
function psdownload(url, to = tempname())
	run("powershell (new-object system.net.webClient).downloadFile(\"$url\", \"$to\")")
	return to
end

export rsync
function rsync(src::String, dst::String, port = 22; zip = true, delete = true, exclude = false, update = false)
	src = is_windows() && cygdrive(src)
  opts = `-avPh`
  zip && (opts = `$opts -z`)
  delete && (opts = `$opts --delete-after`)
  update && (opts = `$opts -u`)
  exclude && (opts = `$opts --cvs-exclude`)
  @ignore run(`rsync $opts -e "ssh -p $port" $src $dst`)
  # a-all v-verpose u-update z-zip P-progress h-humanreadable
end

function rsync(srcs::Array{String}, dsts::Array{String}, port = 22; kwargs...)
  pmap(srcs, dsts) do src, dst
    rsync(src, dst, port; kwargs...)
  end
end

function rsync(srcs::Array{String}, dst::String, port = 22; kwargs...)
  pmap(srcs) do src
    rsync(src, dst, port; kwargs...)
  end
end

function rsync(src::String, dsts::Array{String}, port = 22; kwargs...)
  pmap(dsts) do dst
    rsync(src, dst, port; kwargs...)
  end
end

macro plots()
	ex = :(import Plots)
	if isdefined(:IJulia)
		ex = Expr(:block, ex,
		:(Plots.default(size = (600, 300),
      html_output_format = "png")))
	end
	esc(ex)
end

Base.run(str::AbstractString) = @static is_windows() ? ps(str) : bash(str)

export @bat_str
macro bat_str(str, exe = "run")
  :(bat($str, $(symbol(exe))))
end

export bat
function bat(str, exe = run)
  fn = tempname() * ".bat"
  write(fn, str)
  exe(`$fn`)
end

export @ps_str
# ps"""
# $x = 1
# echo $x
# """
macro ps_str(str, exe = "run")
  :(ps($str, $(symbol(exe))))
end

export ps
# str = """
# \$x = 1
# echo \$x
# """ |> ps
function ps(str, exe = run)
  exe(`powershell -Command $str`)
end

export @bash_str
# bash"""
# ls
# echo $PATH
# python
# """
macro bash_str(str, exe = "run")
  :(bash($str, $(symbol(exe))))
end

export bash
# str = """
# ls
# echo \$PATH
# python
# """ |> bash
function bash(str, exe = run)
  exe(`bash -c $str`)
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

function generate(pkgname)
  pkgname = "Documenter"
  path = Pkg.dir(pkgname, "docs", "make.jl")
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
    vf[i] = @eval $x->$(f.args[i])
  end
  return @eval ($x)->($(vf)[findfirst($c)])($x)
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
        y[I[s]] += A[s] * xc
    end
  end
end

export exreplace!, exreplace
exreplace(ex, r, s) = (ex = deepcopy(ex); exreplace!(ex, r, s))
exreplace!(ex, r, s) = (ex == r) ? s : ex
function exreplace!(ex::Expr, r, s)
  for i in 1:length(ex.args)
      ex.args[i] = exreplace(ex.args[i], r, s)
  end
  ex
end

export typreplace!, typreplace
typreplace(ex, r, s) = (ex = deepcopy(ex); typreplace!(ex, r, s))
typreplace!(ex, r, s) = (typeof(ex) == r) ? exreplace(s, :ex, ex) : ex
function typreplace!(ex::Expr, r, s)
  for i in 1:length(ex.args)
      ex.args[i] = typreplace!(ex.args[i], r, s)
  end
  ex
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
  nr, nc = size(seq)
  stack = zeros(T, nr * tstack, nc)
  for t in 1:nc
    ind = ((t - tstack) * nr + 1):(t * nr)
    for i in 1:size(stack, 1)
      stack[i, t] = ind[i] > 0 ? seq[ind[i]] : seq[mod1(i, nr)]
    end
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
function splitdata(x, y; splitratio = 0.2, featuredim = "col")
	if featuredim == "col"
		getfun, setfun, countfun = cview, cset!, ccount
	elseif featuredim == "row"
		getfun, setfun, countfun = rview, rset!, rcount
	end
  n = countfun(y)
  s = round(Int, (1 - splitratio) * n)
  xtrn, xtst = getfun(x, 1:s), getfun(x, (s + 1):n)
  ytrn, ytst = getfun(y, 1:s), getfun(y, (s + 1):n)
  return  xtrn, ytrn, xtst, ytst
end

export indmax
"indmax(x, y, z)"
function Base.indmax(xs...) # mimic max(xs...)
  Int[indmax(collect(x)) for x in zip(xs...)]
end

"indmax(x, dim)"
function Base.indmax(x, dim::Int) # mimic maximum(x, dim)
  ind2sub(size(x), vec(findmax(x, dim)[2]))[dim]
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


"""
    name = "/tmp/tmp1.txt"
    attach = ["/tmp/tmp1.txt","/tmp/tmp2.txt"]
    mail(name)
    mail(name, attach)
"""
function mail(name)
  spawn(pipeline(`cat $name`,`mail -s "Computation Results" luyaocns@gmail.com`))
end
function mail(name, attach)
  spawn(pipeline(`cat $name`,`mail -s "Computation Results" --attach=$attach luyaocns@gmail.com`))
end

# export @save_output
# macro save_output(ex)
#     quote
#         originalSTDOUT = STDOUT
#         (outRead, outWrite) = redirect_stdout()
#         $(esc(ex))
#         close(outWrite)
#         data = String(readavailable(outRead))
#         close(outRead)
#         redirect_stdout(originalSTDOUT)
#         println(data)
#         open("/tmp/temp.txt","w") do fh
#             write(fh, "Subject: Terminal Email Send\n\n")
#             write(fh, data)
#         end
# 	      spawn(`bash /home/hdd1/YaoLu/Backup/sendmail.sh`)
#     end
# end

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

# import Base: +, -
# +(f::Function, a::Number) = x->(f(x) + a)
# -(f::Function) = x->-f(x)

#
# export ps
# export @ps_str
# function ps(str)
# 	file = tempname()*".ps1"
#   write(file, str)
#   run(`powershell $file`)
# end
# macro ps_str(str)
# 	ps(str)
# end
#
# export @bat_str
# macro bat_str(str)
#   file = tempname()*".bat"
#   write(file, str)
#   run(`$file`)
# end
#
# export @bash_str
# macro bash_str(str)
#   str = replace(str,"\\\$", "\$")
#   file = tempname()*".bash"
#   write(file, str)
#   run(`bash $file`)
# end


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


end # End of Utils
