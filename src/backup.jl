function rename_borg_youtube()
    for (root, dirs, files) in walkdir(pwd())
        for file in files
            fn = joinpath(root, file)
            fn_new = replace(replace(replace(fn, r" \(.*\)", ""), "%3", " "), "_hd", "")
            try mv(fn, fn_new) end
        end
    end
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

macro save(name, ex)
    name = string(name)
    quote
        open("/tmp/$($name).txt","a") do f
            println(f, $(string(ex)))
            println(f, $(esc(ex)))
        end
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


using Base.Threads

function threadcall(f::Function, run_on_thread::Int, args...; kwargs...)
    @assert run_on_thread!=1
    rr = Future()

    function inner()
        function fun()
            if Base.Threads.threadid()==1
                yield()
            end
            if Base.Threads.threadid()!=run_on_thread
                return
            end

            ret = f(args...;kwargs...)
            put!(rr, ret)
        end
        ccall(:jl_threading_run, Nothing, (Any,), Core.svec(fun))  #Run it on all threads
        rr
    end
    @async inner()
    rr
end

export spsum2!
function spsum2!(W, fire, g)
  j=1
  spike = fire[j]
  for i in eachindex(W.nzval)
    if j < W.n && i > W.colptr[j+1]-1; j += 1; spike = fire[j];  end
    if spike; g[W.rowval[i]] += W.nzval[i]; end
  end
end

export @export_all
macro export_all(ex)
  ex_exp = Expr(:block)
  for arg in ex.args
      if arg.head == :(=)
          push!(ex_exp.args,:(export $(arg.args[1])))
      end
      push!(ex_exp.args,arg)
  end
  esc(ex_exp)
end


export @fast
macro fast(ex)
  :(@fastmath(@inbounds($ex)))
end

export @param
macro param(ex)
  bounds = []
  for arg in ex.args[3].args
    if arg.head != :line
      arg1 = arg.args[2]
      if isdefined(arg1, :args)
        push!(bounds, arg1.args[2])
        arg.args[2] = arg1.args[1]
      else
        push!(bounds, arg1)
      end
    end
  end
  push!(ex.args[3].args, :(bounds::Tuple=$(Expr(:tuple, bounds...))))
  esc(:(@with_kw $ex))
end

function histy(y)
    y = vec2y(y)
    if !isdefined(:nclass)
        @eval nclass = $(Int(maximum(y))+1)
    end
    freq =  hist(vec(y),-0.5:1:nclass-0.5)[2]
    prob = freq/sum(freq)
end
using Gadfly, xFrames
function imbalance_view(train_Y, test_Y, preds)
    xticks = Guide.xticks(ticks = [0:nclass-1;])
    yticks = Guide.yticks(ticks = [0:nclass-1;])
    df = xFrame(
        class = test_Y,
        predict = preds
    )
    plot(df, x = :class, y = :predict, Geom.violin, xticks, yticks,
    Guide.title("Error Violin")) |> display
    df = xFrame(
        class = 0:nclass-1,
        percent = histy(test_Y[test_Y .!= preds])
    )
    plot(df, x = :class, y = :percent, Geom.bar, xticks,
    Guide.title("Error Contribution")) |> display
    df_trn = xFrame(
        class = 0:nclass-1,
        percent = histy(train_Y),
        group = "train"
    )
    df_tst = xFrame(
        class = 0:nclass-1,
        percent = histy(test_Y),
        group = "test"
    )
    df_prd = xFrame(
        class = 0:nclass-1,
        percent = histy(preds),
        group = "predict"
    )
    df = vcat(df_trn, df_tst, df_prd)
    plot(df, x = :class, y = :percent, color = :group,
    Geom.bar(position=:dodge), xticks,
    Guide.title("Distribution")) |> display
end


module PlotUtils

export imshow
"imshow(rand(100,100))"
function imshow(A)
    @eval using Images,Colors
    A = (A - minimum(A))./(maximum(A) - minimum(A))
    convert(Image{Gray},A)
end

"""
    using Gadfly
    mplot(x=1:10,y=rand(10,3),Gadfly.Geom.line)
"""
function mplot(o1...;x=[], y=[], o2...)
    @eval using DataFrames
    df = DataFrame()
    for j = 1:size(y,2)
        df = vcat(df, DataFrame(x=x,y=y[:,j],label="$j"))
    end
    Gadfly.plot(df, x=:x, y=:y, color=:label,o1...,o2...) |> display
end

end

import Base: +, -
+(f::Function, a::Number) = x->(f(x) + a)
-(f::Function) = x->-f(x)


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


function smotetomek(x, y)
  SMOTETomek = pyimport("imblearn.combine")[:SMOTETomek]()
  for t in 1:length(unique(y)) - 1
    x, y = SMOTETomek[:fit_sample](x, y)
  end
  return x, y
end

function smote(x, y)
  SMOTE = pyimport("imblearn.over_sampling")[:SMOTE]()
  for t in 1:length(unique(y)) - 1
    x, y = SMOTE[:fit_sample](x, y)
  end
  return x, y
end

function nearmiss(x, y)
  NearMiss = pyimport("imblearn.under_sampling")[:NearMiss](
    version = 2, random_state = 42)
  NearMiss[:fit_sample](x, y)
end

function tomeklinks(x, y)
  TomekLinks = pyimport("imblearn.under_sampling")[:TomekLinks]()
  for t in 1:length(unique(y)) - 1
    x, y = TomekLinks[:fit_sample](x, y)
  end
  return x, y
end

export balance
"""
    x = rand(200, 3); y = rand(1:3, 200)
    xb, yb = balance(x, y)

    x = rand(3, 200); y = rand(1:3, 200)
    xb, yb = balance(x, y)
"""
function balance(x, y, method = smotetomek)
  if size(x, 2) > size(x, 1)
    xb, yb = method(x', vec(y))
    return xb', yb
  else
    return method(x, vec(y))
  end
end

@eval Knet begin
    function conv4{T}(w::Array{T,4}, x::Array{T,4};
        padding = 0, stride = 1, upscale = 1, mode = 0, alpha = 1,
        o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
        if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
        if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
        Wx, Hx, Cx, Nx = size(x)
        Ww, Hw, C1, C2 = size(w)
        if Cx != C1; throw(DimensionMismatch()); end
        Wy, Hy, Cy, Ny = cdims(w, x; padding = padding, stride = stride)
        y = similar(x, (Cy, Wy * Hy * Ny))
        xcol = similar(x, (Wy * Hy * Ny, Ww * Hw * C1))
        x2 = similar(x, (Wy * Hy, Ww * Hw * C1))
        (p1, p2) = psize(padding, x)
        (s1, s2) = psize(stride, x)
        alpha, beta = T(alpha), T(0)
        @inbounds for n in 1:Nx
            # im2col!(w, x, x2, n, p1, p2, s1, s2, mode)
            im2col!(x, n, x2, Wx, Hx, Cx, (Ww, Hy), (p1, p2), (s1, s2))
            ind = (n - 1) * Wy * Hy + 1 : n * Wy * Hy
            xcol[ind, :] = x2
        end
        gemm!('T','T', alpha, reshape(w, Ww * Hw * Cx, :), xcol, beta, y)
        y = permutedims(reshape(y, Cy, Wy, Hy, Ny), [2, 3, 1, 4])
        return y
    end

    function im2col!{T}(img::Array{T}, n::Int, col::Array{T}, width::Int, height::Int, channels::Int,
        kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

        im2col_impl!(view(img,:,:,:,n), col, width, height, channels, kernel, pad, stride)
    end

    function im2col_impl!{T}(img::AbstractArray{T}, col::Array{T}, width::Int, height::Int, channels::Int,
        kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

        kernel_w, kernel_h = kernel
        pad_w, pad_h = pad
        stride_w, stride_h = stride

        height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
        width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
        channels_col = channels * kernel_h * kernel_w

        for c = 0:channels_col-1
            w_offset = c % kernel_w
            h_offset = div(c, kernel_w) % kernel_h
            c_im = div(c, kernel_h * kernel_w) # channel
            for h = 0:height_col-1
                for w = 0:width_col-1
                    h_pad = h*stride_h - pad_h + h_offset
                    w_pad = w*stride_w - pad_w + w_offset
                    if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                        @inbounds col[1 + (c*height_col+h) * width_col + w] =
                        img[1 + (c_im * height + h_pad) * width + w_pad]
                    else
                        @inbounds col[1 + (c*height_col+h) * width_col + w] = 0
                    end
                end
            end
        end
    end
end
