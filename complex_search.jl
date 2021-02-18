using Distributed
addprocs(8)
@everywhere include("reproducibility.jl")
include("reproducibility.jl")

@everywhere using LightGraphs
@everywhere using NPZ
@everywhere using LinearAlgebra
@everywhere using Expokit

##

@everywhere function normalized_laplacian(g::Graph, T)
    m = laplacian_matrix(g, T)
    d = diagm(1. ./ sqrt.(diag(m)))
    d*m*d
end

@everywhere function get_hamiltonian(g::Graph, mode::Symbol)
    m = eval(mode)(g, Float64)
    if mode != :adjacency_matrix
        m *= -1
    end
    Symmetric(Matrix{Float64}(m))
end

@everywhere function normalize_eigen(eigdec, h)
    n = length(eigdec.values)
    λn, λ1 = eigdec.values[[1,n]]
   
    eigdec.values .-= λ1
    eigdec.values ./= λ1 - λn
    eigdec.values .+= 1

    eigdec, (h - λ1*I)/(λ1 - λn) + I
end

@everywhere function true_probability(h, eigdec, node)
    s1 = Sk(eigdec, node, 1)
    s2 = Sk(eigdec, node, 2)

    the_hamiltonian = s1*h 
    the_hamiltonian[node,node] += 1
    init_state = eigdec.vectors[:,end]
  
    time = 1/sqrt(ε(eigdec, node))*sqrt(s2)/s1
    f_state = expmv(time, 1im*the_hamiltonian, init_state)
    abs2(f_state[node])
end

@everywhere function Δ(eigdec)
    eigdec.values[end] - eigdec.values[end-1]
end

@everywhere function Sk(eigdec, node::Int, k::Int)
    n = length(eigdec.values)
    @assert 1 <= k <= n
    @assert 1 <= node <= n
    a = eigdec.vectors[node,1:end-1]
    sum(abs2.(a) ./ (1 .- eigdec.values[1:end-1]).^k)
end

@everywhere function ε(eigdec, node::Int)
    abs2(eigdec.vectors[node,end])
end

# h = get_hamiltonian(erdos_renyi(10, 0.5), :adjacency_matrix)
# println(eigvals(h))

@everywhere function condition(g::SimpleGraph, node::Int, mode::Symbol)
    h = get_hamiltonian(g, mode)
    eigdec, h = normalize_eigen(eigen(h), h)


    s1, s2, s3 = (k->Sk(eigdec, node, k)).([1,2,3])
    my_ε = ε(eigdec, node)
    my_Δ = Δ(eigdec)
    result = Float64[]
    push!(result, sqrt(my_ε)/min(s1*s2/s3, my_Δ*sqrt(s2)))
    push!(result, my_Δ)
    push!(result, abs2(s1)/s2) # probability
    push!(result, 1/sqrt(my_ε)*sqrt(s2)/s1) # time
    push!(result, true_probability(h, eigdec, node))

    result
end


##

data_n = 100:100:5000
repeating = 200
@everywhere linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y

@sync @distributed for m=1:repeating
    println("######### m = $m #########")
    for mode = [:normalized_laplacian, :adjacency_matrix, :laplacian_matrix]
        for vertex = [:1, :n]
            for m0 = 3:3
                @time begin
                    filename = "complex_search/BA-$m0-$vertex-$(string(mode)[1:3])-$m.npz"
                    @show filename
                    
                    g = complete_graph(m0)
                    f = g->condition(g, vertex == :1 ? 1 : nv(g) , mode)

                    data_y = Vector{Float64}[]
                    for n = data_n
                        barabasi_albert!(g, n, m0)
                        push!(data_y, f(g))
                    end

                    npzwrite(filename, hcat(data_n, transpose(hcat(data_y...))))
                end
            end
        end
    end
end

