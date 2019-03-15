export @imports, @from

macro imports(lib, as, abbrev)
    esc(:($abbrev = pyimport($(string(lib)))))
end

macro imports(libs)
    libs = isa(libs, Expr) ? libs.args : [libs]
    exs = [:($lib = pyimport($(string(lib)))) for lib in libs]
    esc(Expr(:block, exs...))
end

macro from(lib, imports, fs)
    lib, fs = string(lib), isa(fs, Expr) ? fs.args : [fs]
    exs = [:($f = pyimport($lib).$(string(f))) for f in fs]
    esc(Expr(:block, exs...))
end
