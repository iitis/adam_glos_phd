using Distributed
addprocs(6)
@everywhere include("reproducibility.jl")
include("reproducibility.jl")
##

@everywhere using QSWalk
@everywhere using LightGraphs
@everywhere using NPZ
@everywhere using LinearAlgebra, SparseArrays
data_folder_name = "convergence_qsw"

@everywhere function connected_ER(n::Int, p::Float64; is_directed::Bool=false)
    g = erdos_renyi(n, p, is_directed=is_directed)
    while !is_weakly_connected(g)
        g = erdos_renyi(n, p, is_directed=is_directed)
    end
    g
end

function distances(g, targets::Vector{Int})
    [minimum(length.([a_star(g, v, t) for t=targets])) for v=1:nv(g)]
end

## #########################################################################
# ##################### structure observance ###############################
# ##########################################################################


ωs = 0.5:0.05:1.
t_inf = 10_000
repeating = 50
n = 15

println("Path LQSW")
# path_graph LQSW
filename = "$data_folder_name/local_structure_observance_path_$n.npz"
if isfile(filename)
    println("$filename exists -- skipped")
else
    data = zeros(length(ωs), n)
    a = diagm(1 => ones(n-1))    
    l = local_lind(a)
    h = diagm(1 => ones(n-1), -1 => ones(n-1))
    init_state = sparse(diagm(fill(1/n, n)))
    for (ωind, ω) = enumerate(ωs)
        s = evolve_generator(h, l, ω)
        data[ωind, :] = abs.(diag(evolve(s, init_state, t_inf)))
    end
    npzwrite(filename, data)
end

# BA LQSW
for m0=[1, 2]
    println("BA $m0 LQSW")
    local filename = "$data_folder_name/local_structure_observance_$n-$m0.npz"
    if isfile(filename)
        println("$filename exists -- skipped")
        continue
    end

    local data = zeros(repeating, length(ωs)+1, n) # +1 for storing distances
    for m=1:repeating
        @show m
        g = barabasi_albert(n, m0, complete=true, is_directed=true)
            
        l = local_lind(adjacency_matrix(g, dir=:in))
        h = adjacency_matrix(Graph(g))
        init_state = sparse(diagm(fill(1/n, n)))
        for (ωind, ω) = enumerate(ωs)
            s = evolve_generator(h, l, ω)
            data[m, ωind, :] = abs.(diag(evolve(s, init_state, t_inf)))
        end 
        data[m, end, :] = distances(g, collect(1:m0))
    end
    npzwrite(filename, data)
end

# path_graph GQSW
println("Path GQSW")
filename = "$data_folder_name/global_structure_observance_path_$n.npz"
if isfile(filename)
    println("$filename exists -- skipped")
else
    data = zeros(length(ωs), n)
    l = diagm(1 => ones(n-1))
    h = diagm(1 => ones(n-1), -1 => ones(n-1))
    init_state = sparse(diagm(fill(1/n, n)))
    for (ωind, ω) = enumerate(ωs)
        s = evolve_generator(h, [l], ω)
        data[ωind, :] = abs.(diag(evolve(s, init_state, t_inf)))
    end
    npzwrite(filename, data)
end

# BA GQSW
for m0=[1, 2]
    println("BA $m0 GQSW")

    local filename = "$data_folder_name/global_structure_observance_$n-$m0.npz"
    if isfile(filename)
        println("$filename exists -- skipped")
        continue
    end

    local data = zeros(repeating, length(ωs)+1, n) # +1 for storing distances
    for m=1:repeating
        @show m
        g = barabasi_albert(n, m0, complete=true, is_directed=true)
            
        l = [adjacency_matrix(g, dir=:in)]
        h = adjacency_matrix(Graph(g))
        init_state = sparse(diagm(fill(1/n, n)))
        for (ωind, ω) = enumerate(ωs)
            s = evolve_generator(h, l, ω)
            data[m, ωind, :] = abs.(diag(evolve(s, init_state, t_inf)))
        end 
        data[m, end, :] = distances(g, collect(1:m0))
    end
    npzwrite(filename, data)
end

# path_graph NGQSW
println("Path NGQSW")
n = 15
filename = "$data_folder_name/nonmoral_structure_observance_path_$n.npz"
if isfile(filename)
    println("$filename exists -- skipped")
else
    local data = zeros(length(ωs), n)
    a = diagm(1 => ones(n-1))
    l, vset = nm_lind(a)
    h = nm_glob_ham(a)
    hrot = nm_loc_ham(vset)
    init_state = nm_init(vlist(vset), vset)
    for (ωind, ω) = enumerate(ωs)
        s = evolve_generator(h, [l], hrot, ω)
        data[ωind, :] = abs.(nm_measurement(evolve(s, init_state, t_inf), vset))
    end
    npzwrite(filename, data)
end

# BA NGQSW
function random_rot(kmax::Int)
    result = Dict{Int,AbstractMatrix{ComplexF64}}()
    for k=1:kmax
        h = rand(k,k)
        result[k] = h + transpose(h)
        h = rand(k,k)
        result[k] += 1im*(h - transpose(h))
    end
    result
end
    
for m0=[1, 2]
    println("BA $m0 NGQSW")
    local filename = "$data_folder_name/nonmoral_structure_observance_$n-$m0.npz"
    if isfile(filename)
        println("$filename exists -- skipped")
        continue
    end

    local data = zeros(repeating, length(ωs)+1, n) # +1 for storing distances
    for m=1:repeating
        @show m
        g = barabasi_albert(n, m0, complete=true, is_directed=true)
        a = adjacency_matrix(g, dir=:in)
        l, vset = nm_lind(a)
        h = nm_glob_ham(a)
        hrot = nm_loc_ham(vset, random_rot(n))
        init_state = nm_init(vlist(vset), vset)
        data[m, end, :] = distances(g, collect(1:m0))
        for (ωind, ω) = enumerate(ωs)
            s = evolve_generator(h, [l], hrot, ω)
            data[m, ωind, :] =  abs.(nm_measurement(evolve(s, init_state, t_inf), vset))
        end        
    end
    npzwrite(filename, data)
end

## #########################################################################
# ############################# NMGQSW probs ###############################
# ##########################################################################
println("Convergence in probs")

n = 12
models = []
push!(models, (n -> barabasi_albert(n, 1, is_directed=true), "ba_1_directed"))
push!(models, (n -> barabasi_albert(n, 1), "ba_1"))
push!(models, (n -> barabasi_albert(n, 3, complete=true), "ba_3"))
push!(models, (n -> barabasi_albert(n, 3, complete=true, is_directed=true), "ba_3_directed"))
push!(models, (n -> connected_ER(n, 0.4, is_directed=true), "er_0.4_directed"))
push!(models, (n -> connected_ER(n, 0.4), "er_0.4"))

t_step = 100
t_count = 100
repeating = 500

@everywhere function random_s(g)
    a = adjacency_matrix(g, dir=:in)
    l, vset = nm_lind(a)
    h = nm_glob_ham(a)
    hrot = nm_loc_ham(vset)
    s = Matrix(evolve_generator(h, [l], hrot))
    s, vset
end

@everywhere function data_gen(m, generator, n, t_count, t_step)
    @show m
        
    s, vset = random_s(generator(n))
    @time v = eigvals(s)
    list_imaginary = sort(imag.(collect(filter(x->abs(real(x)) < 1e-10 && abs(imag(x)) > 1e-10 , v))), by = x->abs(x))
    while length(list_imaginary) == 0
        s, vset = random_s(generator(n))
        @time v = eigvals(s)
        list_imaginary = sort(imag.(collect(filter(x->abs(real(x)) < 1e-10 && abs(imag(x)) > 1e-10 , v))), by = x->abs(x))
    end
    @show size(s, 1)
    flush(stdout)

    @time begin
        u = evolve_operator(s, t_step)
        rho = nm_init(vlist(vset), vset)
        probs = [nm_measurement(rho, vset)]
        @time for _ = 1:t_count
            rho = evolve(u, rho)
            push!(probs, nm_measurement(rho, vset))
        end
    end
    flush(stdout)
    hcat(probs...)
end

for (generator, gen_label) = reverse(models)
    println("########################### $gen_label ###########################")
    data = pmap(m -> data_gen(m, generator, n, t_count, t_step), 1:repeating)

    npzwrite("$data_folder_name/nonmoral_probs_$gen_label-$n.npz", cat(data..., dims=3))
end

