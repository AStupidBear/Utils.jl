function getjobid(cmd)
    println(string(cmd)[2:end-1])
    jobstr = readstring(cmd)
    jobid = match(r"<(\d+)>", jobstr).captures[1]
end

bkill(ids) = for id in ids try run(`bkill $id`) end end

export bsub
function bsub(;queue = "smallopa", subqueue = queue, n = 28, fn = "main")
    jobid = getjobid(`bsub -q $queue -n $n -oo $fn.log -eo $fn.err echo`)
    local nodes
    while true
        jobs = readstring(`bjobs $jobid`)
        nodes = [m.match for m in eachmatch(r"node(\d+)", jobs)]
        nodes = setdiff(nodes, [readchomp(`hostname`)])
        !isempty(nodes) && break
        sleep(1)
    end
    write("$(fn)_compile.jl", readline("$fn.jl"), "using MPI; exit()")
    jobids = [getjobid(`bsub -q $subqueue -m $node -eo $node.err julia $(fn)_compile.jl`) for node in nodes]
    w = join(["done($jobid)" for jobid in jobids], "&&")
    nodes = join(nodes, " ")
    run(`bsub -q $queue -n $n -w $w -m $nodes -oo $fn.log -eo $fn.err -J $fn mpijob-new julia $fn.jl`)
end
