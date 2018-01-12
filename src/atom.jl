enable(pkg) = try run(`apm.cmd enable $pkg`) end

disable(pkg) = try run(`apm.cmd disable $pkg`) end

function toggle(env_name, pkgs)
    if !haskey(ENV, env_name)
        run(`setx $env_name 0`)
    elseif ENV[env_name] == "1"
        disable.(pkgs)
        run(`setx $env_name 0`)
    elseif ENV[env_name] == "0"
        enable.(pkgs)
        run(`setx $env_name 1`)
    end
end
