export Parameter, getparam, catparam, setparam!, delete_line!, unroll, unroll_block!
export @rep, @param, @withkw

abstract type Parameter end

getparam(param::Parameter) = fieldvalues(param)[1:(end - 1)]

catparam(param::T, c) where {T <: Parameter} = T(c..., param.bounds)

function setparam!(param::Parameter, name, x)
    setfield!(param, name, x)
    i = findfirst(x -> x == name, fieldnames(param))
    bounds = param.bounds
    if length(bounds[i]) <= 1 || (bounds[i] <: AbstractString)
        param.bounds = (bounds[1:i-1]..., x, bounds[i+1:end]...)
    end
    return param
end

function Base.display(param::Parameter)
    println("Parameters: ")
    for s in fieldnames(param)
        s != :bounds && println(s, "=", getfield(param, s))
    end
end

macro rep(n, ex)
    name = ex.args[1].args[1] |> string
    exs = [exreplace(ex, Symbol(name), Symbol(name*"$i")) for i in 1:n]
    esc(Expr(:block, exs...))
end

macro withkw(ex)
    ex = macroexpand(ex)
    :(Parameters.@with_kw $ex) |> esc
end

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

function Base.insert!(collection::Vector{T}, index::Integer, items::Vector{T}) where T
    for item in items
        Base.insert!(collection, index, item)
        index += 1
    end
    collection
end

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

unroll(x, name) = [getfield(x, s) for s in fieldnames(x) if occursin(string(name), string(s))]
