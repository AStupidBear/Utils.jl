macro curry(n, f)
    syms = [gensym() for i=1:n]
    foldl((ex, sym)  -> Expr(:->, sym, ex), Expr(:call, f, syms...), reverse(syms))
end

export partial
partial(f, a...; b...) = ((x...; y...) -> f(a..., x...; b..., y...))
