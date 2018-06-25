macro curry(n, f)
    syms = [gensym() for i=1:n]
    foldl((ex, sym)  -> Expr(:->, sym, ex), Expr(:call, f, syms...), reverse(syms))
end

export partial
partial(f, a...; b...) = ((x...; y...) -> f(a..., x...; b..., y...))

export throttle
function throttle(f, timeout; leading = true, trailing = false)
  lasttime = time()
  function throttled(args...; kwargs...)
    result = nothing
    if time() > lasttime + timeout
        result = f(args...; kwargs...)
        lasttime = time()
    end
    return result
  end
end