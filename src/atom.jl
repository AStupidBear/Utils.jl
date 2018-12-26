using Atom: isuntitled, isfile′, basepath, pkgpath

function Atom.expandpath(path)
    name, path = if isempty(path)
        (path, path)
    elseif occursin(r"\./[A-F]", path)
        (basename(path), replace(path, "./" => ""))
    elseif path == "./missing"
        ("<unknown file>", path)
    elseif isuntitled(path)
        ("untitled", path)
    elseif !isabspath(path)
        (normpath(joinpath("base", path)), basepath(path))
    elseif occursin(joinpath("julia", "stdlib"), path)
        p = last(split(path, joinpath("julia", "stdlib", "")))
        name = normpath(joinpath("stdlib", p))
        path = isfile′(path) ? path : normpath(joinpath(basepath(joinpath("..", "stdlib")), p))
        name, path
    else
        (pkgpath(path), path)
    end
    path = replace(path, homedir() => "D:")
    return name, path
end

# enable(pkg) = try run(`apm.cmd enable $pkg`) end
#
# disable(pkg) = try run(`apm.cmd disable $pkg`) end
#
# function toggle(env_name, pkgs)
#     if !haskey(ENV, env_name)
#         run(`setx $env_name 0`)
#     elseif ENV[env_name] == "1"
#         disable.(pkgs)
#         run(`setx $env_name 0`)
#     elseif ENV[env_name] == "0"
#         enable.(pkgs)
#         run(`setx $env_name 1`)
#     end
# end
