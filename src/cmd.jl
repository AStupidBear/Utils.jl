Base.run(str::AbstractString) = @static Sys.iswindows() ? ps(str) : bash(str)

export @bat_str
macro bat_str(str, exe = "run")
    :(bat($str, $(symbol(exe))))
end

export bat
function bat(str, exe = run)
    fn = tempname() * ".bat"
    write(fn, str)
    exe(`$fn`)
end

export @ps_str
# """
# ```
# ps"""
# $x = 1
# echo $x
# """
# ```
# """
macro ps_str(str, exe = "run")
    :(ps($str, $(symbol(exe))))
end

export ps
# """
# ```
# str = """
# \$x = 1
# echo \$x
# """ |> ps
# ```
# """
function ps(str, exe = run)
    exe(`powershell -Command $str`)
end

export @bash_str
# ```
# """
# bash"""
# ls
# echo $PATH
# python
# """
# """
# ```
macro bash_str(str, exe = "run")
    :(bash($str, $(symbol(exe))))
end

export bash
# """
# ```
# str = """
# ls
# echo \$PATH
# python
# """ |> bash
# ```
# """
function bash(str, exe = run)
    exe(`bash -c $str`)
end

export code2cmd
code2cmd(str) = replace(replace(str, "\n",""), "\"", "\\\"")
