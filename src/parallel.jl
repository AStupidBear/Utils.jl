export pmapreduce
pmapreduce(f, op, iter) = reduce(op, pmap(f, iter))

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
        pstree = read(`pstree -s $(getpid())`, String)
        occursin("mpi", pstree) || occursin("slurm", pstree)
    catch
        !iswindows()
    end
end

export scc_start
function scc_start()
    iswindows() && return
    inmpi() && @eval Main begin
        using MPI; mngr = MPI.start_main_loop(MPI.MPI_TRANSPORT_ALL)
    end
end

export scc_end
function scc_end()
    isdefined(Main, :MPI) && @eval Main begin
        MPI.stop_main_loop(mngr)
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
