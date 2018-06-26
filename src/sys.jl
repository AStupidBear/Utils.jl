export memory
memory(x) = Base.summarysize(x) / 1024^2

"""cron("spam.jl", 1)"""
function cron(fn, repeat)
    name = splitext(fn)[1]
    vb = """
    DIM objShell
    set objShell=wscript.createObject("wscript.shell")
    iReturn=objShell.Run("cmd.exe /C $(abspath(fn))", 0, TRUE)
    """
    bat = """
    schtasks /create /tn "$name" /sc minute /mo $repeat /tr "$(abspath("$name.vbs"))"
    schtasks /run /tn "$name"
    """
    write("$name.vbs", vb)
    write("task.bat", bat)
    run(`task.bat`)
end

export parseenv
parseenv(key, default) = parse(typeof(default), get(ENV, string(key), string(default)))
parseenv(key, default::String) = get(ENV, string(key), string(default))

# function proxy(url)
#     regKey = "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings"
#     run(`powershell Set-ItemProperty -path \"$regKey\" AutoConfigURL -Value $url`)
# end
# cow() = proxy("http://127.0.0.1:7777/pac")

# function linux_backup(dir = "/home/hdd1/YaoLu/Software", user = "luyao")
#     date = string(now())[1:10]
#     sysfile = joinpath(dir, "$date-sys.tar.gz")
#     run(`sudo tar czf $file --exclude=/home --exclude=/media --exclude=/dev --exclude=/mnt --exclude=/proc --exclude=/sys --exclude=/tmp --exclude=/run /`)
#     userfile = joinpath(dir, "$date-$user.tar")
#     run(`sudo 7z a $userfile /home/$user`)
# end
#
# function linux_restore(file)
#     run(`tar xf $(abspath(file)) -C /`)
# end
