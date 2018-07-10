export pmapreduce
pmapreduce(f, op, iter) = reduce(op, pmap(f, iter))

export @everynode
macro everynode(ex)
    quote
        hosts = @parallel (vcat) for i in workers() gethostname() end
        @sync @parallel for i in 1:nworkers() i == findfirst(x -> x == hosts[i], hosts) && $ex end
    end |> esc
end

export scc_setup
function scc_setup()
    is_linux() && !isinteractive() && @eval current_module() begin
        using MPI; mngr = MPI.start_main_loop(MPI.MPI_TRANSPORT_ALL)
    end
end

export scc_end
scc_end() = @eval current_module() (isdefined(Main, :MPI) && MPI.stop_main_loop(mngr); exit())

export everythread
everythread(fun) = ccall(:jl_threading_run, Ref{Void}, (Any,), fun)

# export aws_setup
# function aws_setup(n = 0)
#     @eval using ClusterManagers; addprocs_qrsh(n)
#     @everynode begin
#         try run(`sudo mkfs -t ext4  /dev/nvme0n1`) end
#         try run(`sudo mkdir /scratch`) end
#         try run(`sudo mount /dev/nvme0n1 /scratch`) end
#         run(`awk 'BEGIN {cmd="sudo cp -ri /shared/Data /scratch/"; print "n" |cmd;}'`)
#         run(`sudo chmod -R ugo+rw /scratch`)
#     end
# end
