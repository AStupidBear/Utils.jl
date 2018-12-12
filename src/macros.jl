export @print
macro print(exs...)
    str = join(["$ex = %.2f " for ex in exs], ", ")
    esc(:(@printf($str, $(expr...))))
end

macro debug(ex)
    x = gensym()
    :($x = $ex; debug($x); $x) |> esc
end

macro info(ex)
    x = gensym()
    :($x = $ex; info($x); $x) |> esc
end

export @repeat
macro repeat(n, ex)
    Expr(:block, (ex for i in 1:n)...) |> esc
end

export @DataFrame
macro DataFrame(exs...) esc(Expr(:call, :DataFrame, Expr(:parameters, Expr.(:call, :(=>), QuoteNode.(exs), exs)...))) end

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
        push!(blk.args, :($ex = getfield($typ, $exquot)))
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
        eval(current_module(), :($(key) = $val))
    end
end

export @undict
"""
```
d = Dict("a"=>[1,2,3], "b" => 2)
@undict d a b
d = Dict(:a=>[1,2,3], :b => 2)
@undict d a b
```
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

export @setdict
macro setdict(d, keys...)
    ex = Expr(:block)
    for s in keys
        push!(ex.args, :($d[$(string(s))] = $s))
    end
    return esc(ex)
end

macro logto(fn)
    quote
        logstream = open(joinpath(tempdir(), $fn * ".log"), "w")
        Logging.configure(output = [logstream, STDOUT], level = Logging.DEBUG)
    end |> esc
end

export @NT
macro NT(xs...)
    @static if VERSION >= v"0.7.0"
        xs = [:($x = $x) for x in xs]
        esc(:(($(xs...),)))
    else
        esc(:(@NT($(xs...))($(xs...))))
    end
end

export @ignore
macro ignore(ex)
    :(try $ex; catch e; warn(e); end) |> esc
end

export @catcherr
macro catcherr(ex)
    :(try $ex; catch e; warn(e); return; end) |> esc
end

export @correct
macro correct(ex)
    res = gensym()
    :($res = try $ex finally end; $res == nothing ? false : $res) |> esc
end

export ntry
macro ntry(ex, n = 1000)
    :(for t in 1:$n
        try $ex; break; catch e; warn(e); end
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

export @dir_str
macro dir_str(src)
    joinpath(dirname(string(__source__.file)), src)
end

export @include
macro include(src)
    src = joinpath(dirname(string(__source__.file)), src)
    :(@eval include($src))
end

export @redirect
macro redirect(src, ex)
    src = src == :devnull ? "/dev/null" : src
    quote
        io = open($(esc(src)), "a")
        o, e = stdout, stderr
        redirect_stdout(io)
        redirect_stderr(io)
        $(esc(ex)); sleep(0.01)
        flush(io); close(io)
        redirect_stdout(o)
        redirect_stderr(e)
    end
end

# using Lazy: isexpr, rmlines, splitswitch
# export @switch
# macro switch(args...)
#     test, exprs = splitswitch(args...)
#
#     length(exprs) == 0 && return nothing
#     length(exprs) == 1 && return esc(exprs[1])
#
#     test_expr(test, val) =
#     test == :_      ? val :
#     has_symbol(test, :_) ? :(let _ = $val; $test; end) :
#     :($test==$val)
#
#     thread(val, yes, no) = :($(test_expr(test, val)) ? $yes : $no)
#     thread(val, yes) = thread(val, yes, :(error($"No match for $test in @switch")))
#     thread(val, yes, rest...) = thread(val, yes, thread(rest...))
#
#     esc(thread(exprs...))
# end
