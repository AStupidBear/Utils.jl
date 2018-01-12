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
