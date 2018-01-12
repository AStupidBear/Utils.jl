export vplot, labelplot, @plots

vplot(ps::Vector) = vplot(ps...)

function vplot(ps...)
    N = length(ps)
    Main.plot(ps..., size = (600, 400N), layout = (N, 1), leg = false)
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

function labelplot(y, label)
    p = Main.plot(;legend = nothing)
    told = 1
    for t in 2:length(y)
        if label[t] != label[t-1] || t == length(y)
            Main.plot!(p, told:t, y[told:t]; color = Int(label[t-1]), label = label[t-1])
            told = t
        end
    end
    return p
end

export @cat
"""
```julia
p = @cat begin
plot(rand(10))
plot(rand(100))
end
@cat [1,2] [1,2,3]
```
"""
macro cat(exs)
    expr = Expr(:block, :(p = []))
    for ex in exs.args
        ex.head != :line && push!(expr.args, :(push!(p, $(esc(ex)))))
    end
    expr
end

macro cat(exs...)
    expr = Expr(:block, :(p = []))
    for ex in exs
        push!(expr.args, :(push!(p, $(esc(ex)))))
    end
    expr
end
