export rsync
function rsync(src::String, dst::String, port = 22; passwd = "", opts = `-avPh`)
    src = Sys.iswindows() ? cygdrive(src) : src
    if isempty(passwd)
        @ignore run(`rsync $opts -e "ssh -p $port" $src $dst`)
    else
        @ignore run(`rsync $opts -e "sshpass -p $passwd ssh -p $port" $src $dst`)
    end
    # a-all v-verpose u-update z-zip P-progress h-humanreadable --cvs-exclude --delete-after
end

export parsync
parsync(src, dst, args...; excludes = [], kwargs...) = rsync(symdiff(readabsdir(src), excludes), dst, args...; kwargs...)

function rsync(srcs::Array, dsts::Array, port = 22; kwargs...)
    pmap(srcs, dsts) do src, dst
        rsync(src, dst, port; kwargs...)
    end
end

function rsync(srcs::Array, dst::String, port = 22; kwargs...)
    pmap(srcs) do src
        rsync(src, dst, port; kwargs...)
    end
end

function rsync(src::String, dsts::Array, port = 22; kwargs...)
    pmap(dsts) do dst
        rsync(src, dst, port; kwargs...)
    end
end


export linux_path, cygdrive

linux_path(path) = replace(path, "\\", "/")

cygdrive(path) = joinpath("/cygdrive/", replace(linux_path(path), ":", ""))

# function junocloud(ip, user, port)
#     local_root = joinpath(homedir(), "Documents", "Codes") |> linux_path
#     remote_root = "/home/$user/Documents"
#
#     m = match(r"connect\((.*)\)", clipboard())
#     remoteport = is(m, nothing) ? "55440" : m[1]
#     localport = "1234"
#
#     julia_eval = """using Juno;Juno.connect(1234)"""
#     ssh_eval = """chmod 400 ~/.ssh/id_rsa; ~/julia-0.5/bin/julia -i -e "$julia_eval"; bash"""
#
#     src= joinpath(homedir(), ".juliarc.jl")
#     dst = "$user@$ip:/home/$user/.juliarc.jl"
#     rsync(src, dst, port)
#
#     src = joinpath(homedir(), ".ssh", "id_rsa")
#     dst = "$user@$ip:/home/$user/.ssh/id_rsa"
#     rsync(src, dst, port)
#
#     src = local_root
#     dst = "$user@$ip:$remote_root"
#     rsync(src, dst, port)
#
#     cmd = `ssh -X -R $localport:localhost:$remoteport -p $port $user@$ip -t $ssh_eval`
#     run(cmd)
# end
