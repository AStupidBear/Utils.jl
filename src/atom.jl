using Atom: Method, HTML

function Atom.view(m::Atom.Method)
    str = sprint(show, "text/html", m)
    str = replace(str, r" in .* at .*$" => "")
    str = string("<span>", str, "</span>")
    tv, decls, file, line = Base.arg_decl_parts(m)
    # file = replace(string(file), "/BIGDATA1/highchain_ylu_1" => "D:")
    link = file == :null ? "not found" : Atom.baselink(string(file), line)
    file = replace(link.file, "/BIGDATA1/highchain_ylu_1" => "Z:")
    HTML(str), Atom.Link(file, link.line, link.contents...)
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
