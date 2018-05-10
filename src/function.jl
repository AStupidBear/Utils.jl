macro curry(n, f)
    syms = [gensym() for i=1:n]
    foldl((ex, sym)  -> Expr(:->, sym, ex), Expr(:call, f, syms...), reverse(syms))
end

export partial
partial(f, a...; b...) = ((x...; y...) -> f(a..., x...; b..., y...))

export throttle
function throttle(f, timeout; leading = true, trailing = false)
  cooldown = true
  later = nothing
  result = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        result = f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @schedule try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> (result = f(args...; kwargs...))
    end

    return result
  end
end