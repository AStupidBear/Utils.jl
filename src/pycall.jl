export @imports, @from

macro imports(lib, as, abbrev)
    esc(:($abbrev = pyimport($(string(lib)))))
end

macro imports(lib)
    exs = []
    if isa(lib, Expr) && lib.head == :tuple
        for arg in lib.args
            var, mdl = arg, string(arg)
            push!(exs, :($var = pyimport($mdl)))
        end
    elseif isa(lib, Expr) && lib.head == :.
        var, mdl = gensym(lib.args[2].value), string(lib)
        mdl = replace(mdl, r"[\(\)]" => "")
        push!(exs, :($var = pyimport($mdl)))
    else
        var, mdl = lib, string(lib)
        push!(exs, :($var = pyimport($mdl)))
    end
    esc(Expr(:block, exs...))
end

macro from(lib, imports, fs)
    lib, fs = string(lib), isa(fs, Expr) ? fs.args : [fs]
    exs = [:($f = pyimport($lib).$(string(f))) for f in fs]
    esc(Expr(:block, exs...))
end
