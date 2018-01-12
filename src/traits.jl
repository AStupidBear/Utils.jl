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
