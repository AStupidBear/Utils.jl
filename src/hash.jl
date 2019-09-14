using Base: SummarySize, SimpleVector

export memoryhash

function memoryhash(obj;
                     exclude = Union{DataType, Core.TypeName, Core.MethodInstance},
                     chargeall = Union{Core.TypeMapEntry, Method})
    @nospecialize obj exclude chargeall
    ss = SummarySize(IdDict(), Any[], Int[], exclude, chargeall)
    size::Int = ss(obj)
    h = UInt(0)
    while !isempty(ss.frontier_x)
        # DFS heap traversal of everything without a specialization
        # BFS heap traversal of anything with a specialization
        x = ss.frontier_x[end]
        i = ss.frontier_i[end]
        val = nothing
        if isa(x, SimpleVector)
            nf = length(x)
            if isassigned(x, i)
                val = x[i]
            end
        elseif isa(x, Array)
            nf = length(x)
            if ccall(:jl_array_isassigned, Cint, (Any, UInt), x, i - 1) != 0
                val = x[i]
            end
        else
            nf = nfields(x)
            ft = typeof(x).types
            if !isbitstype(ft[i]) && isdefined(x, i)
                val = getfield(x, i)
            end
        end
        if nf > i
            ss.frontier_i[end] = i + 1
        else
            pop!(ss.frontier_x)
            pop!(ss.frontier_i)
        end
        if val !== nothing && !isa(val, Module) && (!isa(val, ss.exclude) || isa(x, ss.chargeall))
            h = hash(hash(val), h)
        end
    end
    return h
end