"""
    import PINning
    t = 0:0.1:100
    f = cos(2π*t/50) # target signal
    @time net = PINning.force(t, f'; records=[:z])
    z = Float64[net.records[:z][t][1] for t=1:length(t)];
    plot([f z])
"""
module PINning

import Utils

const g = 1.5 # coupling coeff
const α = 1 # P(0) = αI
export Network
type Network{T}
    No::Int
    Np::Int
    J::Array{T} # normalized recurrent weight
    P::Array{T} # inverse of C = <rr'> dim: Np*Np
    x::Array{T} # current
    r::Array{T} # rate
    z::Array{T} # output dim: No
    q::Array{T} # q = P(t-1)*r dim: Np
    c::T        # c = 1/(1+dot(q,r))
    records::Dict
end

function Network(N, No, Np)
    J = g*1/√N*randn(N,N)
    P = α*eye(N)
    x = 0.5*randn(N)
    r = tanh(x)
    z = 0.5*randn(N)
    q = randn(N)
    c = 0.0
    records = Dict()
    Network(No, Np, J, P, x, r, z, q, c, records)
end

Utils.@replace function train!(net::Network, f, Δt)
    x .+= Δt.*(-x .+ z)
    r .= tanh.(x)
    z .= J*r
    q .= P*r
    c = 1/(1+dot(q,r))
    BLAS.ger!(c, f-@view(z[1:No]), @view(q[1:Np]), @view(J[1:No,1:Np]))
#     BLAS.ger!(c, f-z[1:No], q[1:Np], @view(J[1:No,1:Np]))
    BLAS.ger!(-c, q, q, P)
end

Utils.@replace function sim!(net::Network, f, Δt)
    x .+= Δt.*(-x .+ z)
    r .= tanh.(x)
    z .= J*r
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
        train!(net, f[:,t], Δt)
        record!(net)
    end
    return net
end

export force
function force(t, f; N=1000, Np=1000, No=1, records=[:z])
    net = Network(N, No, Np)
    Δt = t[2] - t[1]
    T = length(t) # duration in the unit of Δt
    monitor(net, records)
    for t = 1:3T÷4
        train!(net, f[:,t], Δt)
        record!(net)
    end
    for t = 3T÷4+1:T
        sim!(net, f[:,t], Δt)
        record!(net)
    end
    return net
end

end
