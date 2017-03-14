"""
    import ForceLearning
    t = 0:0.1:100
    f = cos(2π*t/50) # target signal
    @time = ForceLearning.force(t, f)
    plot([f net.records[:z]])
"""
module ForceLearning

import Utils

const g = 1.5 # coupling coeff
const α = 1 # P(0) = αI
export Network
type Network{T}
    J::Array{T} # normalized recurrent weight
    w::Array{T} # output weight
    u::Array{T} # force weight
    P::Array{T} # inverse of C = <rr'>
    x::Array{T} # current
    r::Array{T} # rate
    z::T        # output
    q::Array{T} # q = P*r
    c::T        # c = 1/(1+dot(q,r))
    records::Dict
end

function Network(N)
    J = g*1/√N*randn(N,N); # normalized recurrent weight
    w = 1/√N*(2*rand(N)-1) # initial output weight
    u = 2*rand(N)-1 # initial force weight
    P = α*eye(N) # initial inverse of C = <rr'>
    x = 0.5*randn(N) # initial current
    r = tanh(x) # initial rate
    z = 0.5*randn() # initial output
    q = zeros(N)
    c = 0.0
    records = Dict()
    Network(J, w, u, P, x, r, z, q, c, records)
end

Utils.@replace function train!(net::Network, f, Δt)
    x .+= Δt.*(-x .+ J*r .+ u*z)
    r .= tanh.(x)
    z = dot(w, r)
    q .= P*r
    c = 1/(1+dot(q,r))
    w .+= c.*(f-z).*q
    BLAS.ger!(-c, q, q, P)
end

Utils.@replace function sim!(net::Network, Δt)
    x .+= Δt.*(-x .+ J*r .+ z*u)
    r .= tanh.(x)
    z = dot(w, r)
end

function record!(net::Network)
    for (key, val) in net.records
        push!(val, copy(getfield(net, key)))
    end
end

export monitor
function monitor(net::Network, syms)
    for sym in syms
        net.records[sym] = Vector{typeof(getfield(net, sym))}()
    end
end

export force!
function force!(net::Network, t, f)
    Δt = t[2] - t[1]
    for t = 1:length(t)
        train!(net, f[t], Δt)
        record!(net)
    end
    return net
end

export force
function force(t, f; N=1000, records=[:z])
    net = Network(N)
    Δt = t[2] - t[1]
    T = length(f) # duration in the unit of Δt
    monitor(net, records)
    for t = 1:3T÷4
        train!(net, f[t], Δt)
        record!(net)
    end
    for t = 3T÷4+1:T
        sim!(net, Δt)
        record!(net)
    end
    return net
end

end
