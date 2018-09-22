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

export fieldvalues, fields
fieldvalues(x) = [getfield(x, s) for s in fieldnames(x)]
fields(x) = [(s, getfield(x, s)) for s in fieldnames(x)]

export hasfield
hasfield(x, s) = isdefined(x, s)

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

export typename
function typename(x::T) where T
    name, ext = splitext(string(T.name))
    isempty(ext) ? name : ext[2:end]
end

export typeparam
typeparam(x::T) where {T} = T.parameters[1]
