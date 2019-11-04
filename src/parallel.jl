export pmapreduce
pmapreduce(f, op, iter) = reduce(op, pmap(f, iter))

export headnodes
function headnodes()
    f = n -> remotecall_fetch(gethostname, n)
    hosts = [(n, f(n)) for n in workers()]
    sort!(hosts, by = reverse)
    nodes = first.(unique(last, hosts))
end

export localworkers
function localworkers()
    f = n -> remotecall_fetch(gethostname, n)
    pool = [n for n in workers() if f(n) == gethostname()]
end

export @everynode
macro everynode(ex)
    quote
        hosts = @parallel (vcat) for i in workers() gethostname() end
        @sync @parallel for i in 1:nworkers() i == findfirst(x -> x == hosts[i], hosts) && $ex end
    end |> esc
end

export inmpi
function inmpi()
    try
        @static if Sys.iswindows()
            occursin("mpi", join(processname.(pstree())))
        else
            ps = read(`pstree -s $(getpid())`, String)
            occursin("mpi", ps) || occursin("slurm", ps)
        end
    catch
        false
    end
end

export scc_start
function scc_start()
    if !inmpi()
        @eval Main macro mpi_do(mgr, expr) esc(expr) end
    else
        @eval Main begin
            using MPI, MPIClusterManagers
            using MPIClusterManagers: @mpi_do
            const MCM = MPIClusterManagers
            mngr = MCM.start_main_loop(MCM.MPI_TRANSPORT_ALL)
        end
    end
end

export scc_end
function scc_end()
    isdefined(Main, :MPI) && @eval Main begin
        MCM.stop_main_loop(mngr); exit()
    end
end

export everythread
everythread(fun) = ccall(:jl_threading_run, Ref{Nothing}, (Any,), fun)

export threadprint
threadprint(x) =  ccall(:jl_, Nothing, (Any,), x)

function get_num_threads() # anonymous so it will be serialized when called
    blas = BLAS.vendor()
    # Wrap in a try to catch unsupported blas versions
    try
        if blas == :openblas
            return ccall((:openblas_get_num_threads, Base.libblas_name), Cint, ())
        elseif blas == :openblas64
            return ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
        elseif blas == :mkl
            return ccall((:MKL_Get_Max_Num_Threads, Base.libblas_name), Cint, ())
        end
        # OSX BLAS looks at an environment variable
        if Sys.isapple()
            return ENV["VECLIB_MAXIMUM_THREADS"]
        end
    finally
    end
    return nothing
end

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
