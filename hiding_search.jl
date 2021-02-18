using Distributed
addprocs(8)
@everywhere include("reproducibility.jl")
include("reproducibility.jl")

##
@everywhere using LightGraphs
@everywhere using LinearAlgebra
@everywhere using SparseArrays
@everywhere using Expokit
using LambertW
using Base.Iterators
using NPZ

## 

@everywhere function hamiltonian_and_normed_eigen(g)
    h = -Matrix{Float64}(laplacian_matrix(g))
    eigdec = eigen(h)
    # eigenvalues are nonnegative
    h ./= abs(minimum(eigdec.values)) 
    eigdec.values ./= abs(minimum(eigdec.values))
    # eigenvalues are in [-1,0]
    eigdec.values .+= 1
    h += I
    # eigenvalues are in [0,1], spectral gap next to 1

    λ1, λ2, λn = eigdec.values[[end, end-1, 1]]
    a = -(λ2 + λn)/2
    eigdec.values .+= a
    h += a*I
    eigdec.values ./= 1 + a
    h ./= 1 + a
    sparse(h), eigdec
end

@everywhere function true_probability(h, eigdec)
    n = size(h, 1)
    a = eigdec.vectors[1,1:end-1]
    eigs = eigdec.values[1:end-1]
    
    gamma = sum(abs2.(a) ./ (1 .- eigs)) / sum(abs2.(a))
    opt_time = pi*sqrt(n)/2
    #opt_time = pi/2/abs(a[1,end]) * sqrt(sum(abs2.(a) ./ (1 .- eigs).^2))

    init_state = fill(1/sqrt(n), n)
    hamiltonian = gamma*h
    hamiltonian[1,1] += 1.
    f_state = expmv(opt_time, -1im*hamiltonian, init_state)
    abs2(f_state[1])
end

@everywhere function create_data(n::Int, p::Float64)
    g = erdos_renyi(n, p)
    cc = connected_components(g)
    g = induced_subgraph(g, cc[findmax(length.(cc))[2]])[1]
    @assert is_connected(g)
    @assert nv(g) > n/2

    h, eigdec = hamiltonian_and_normed_eigen(g)
    c = eigdec.values[end-1]

    (1-c)/(1+c), true_probability(h, eigdec)
end

p0s = .5:.5:2
@everywhere p(p0, n) = p0*log(n)/n
repeating = 50

data_n = 100:100:2000

for p0 = p0s
    data = zeros(length(data_n), repeating, 2)
    for (ind_n, n) = enumerate(data_n)
        println("n $n, p0 $p0")
        @time data_tmp = pmap(_ -> create_data(n, p(p0, n)), 1:repeating)
        for (m, d_tmp) = enumerate(data_tmp)
            data[ind_n,m,:] .= d_tmp
        end
    end
    npzwrite("hiding_search/sparse_ER_probs_and_bounds_$p0.npz", data)
end
rmprocs()
