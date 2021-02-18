using Distributed
addprocs(6)
@everywhere include("reproducibility.jl")
using Random 
Random.seed!(42)
##

@everywhere using QSWalk
@everywhere using LightGraphs
@everywhere using NPZ
@everywhere using LinearAlgebra, SparseArrays
data_folder_name = "convergence_qsw"

## graph_generators

@everywhere function multi_sink_condensation(n::Int, p::Float64)
    g = erdos_renyi(n, p, is_directed=true)
    h = condensation(g)
    while length(filter(v->outdegree(h, v) == 0 , 1:nv(h))) <= 1 || !is_weakly_connected(g)
        g = erdos_renyi(n, p, is_directed=true)
        h = condensation(g)
    end
    g
end

@everywhere function single_sink_condensation(n::Int, p::Float64)
    g = erdos_renyi(n, p, is_directed=true)
    h = condensation(g)
    while length(filter(v->outdegree(h, v) == 0 , 1:nv(h))) > 1
        g = erdos_renyi(n, p, is_directed=true)
        h = condensation(g)
    end
    g
end

@everywhere function random_oriented_tree(n::Int)
    g = DiGraph(n)
    for i=2:n
        if rand() < 0.5
            add_edge!(g, i, rand(1:(i-1)))
        else
            add_edge!(g, rand(1:(i-1)), i)
        end
    end
    g
end

@everywhere function random_orientation(g::Graph)
    digraph = DiGraph(nv(g))
    for e = edges(g)
        i, j = e.src, e.dst
        if rand() < 0.5
            add_edge!(digraph, i, j)
        else
            add_edge!(digraph, j, i)
        end
    end
    digraph
end

@everywhere function random_multi_sink_orientation(n::Int, p::Float64)
    g = random_orientation(erdos_renyi(n, p))
    h = condensation(g)
    while length(filter(v->outdegree(h, v) == 0 , 1:nv(h))) <= 1 || !is_weakly_connected(g)
        g = random_orientation(erdos_renyi(n, p))
        h = condensation(g)
    end
    g
end

@everywhere function multi_sink_condensation_ba(n::Int, k::Int)
    g = random_orientation(barabasi_albert(n, k, complete=true))
    h = condensation(g)
    while length(filter(v->outdegree(h, v) == 0 , 1:nv(h))) <= 1
        g = random_orientation(barabasi_albert(n, k, complete=true))
        h = condensation(g)
    end
    g
end
@everywhere function multi_sink_condensation_tree(n::Int)
    g = random_oriented_tree(n)
    h = condensation(g)
    while length(filter(v->outdegree(h, v) == 0 , 1:nv(h))) <= 1
        g = random_oriented_tree(n)
        h = condensation(g)
    end
    g
end

@everywhere function connected_ER(n::Int, p::Float64; is_directed::Bool)
    g = erdos_renyi(n, p, is_directed=is_directed)
    while !is_weakly_connected(g)
        g = erdos_renyi(n, p, is_directed=is_directed)
    end
    g
end

## #########################################################################
# ############################# GQSW #######################################
# ##########################################################################

gens = [("ba_3", n -> barabasi_albert(n, 3, complete=true, is_directed=true)),
        ("er_0.4", n -> connected_ER(n, 0.4, is_directed=true)),
        ("ba_1", n -> barabasi_albert(n, 1, complete=true, is_directed=true))]

for (gen_lab, gen) = gens, n = 10:10:50
    println("$gen_lab $n")
    filename = "$data_folder_name/global_gaps_$gen_lab-$n.npz"
    if isfile(filename)
        println("File exists - computation skipped")
        continue
    end
    gaps = @sync @distributed (hcat) for m = 1:500
        println("$m")
        g = gen(n)
        l = Matrix(adjacency_matrix(g, dir=:in))
        h = adjacency_matrix(Graph(g))
        s = Matrix(evolve_generator(h, [l]))
        v = eigvals(s)
        second_minimal_abs = sort(abs.(v))[2]
        list_imaginary = sort(imag.(collect(filter(x->abs(real(x)) < 1e-10 && abs(imag(x)) > 1e-10 , v))), by = x->abs(x))
        second_minimal_imag = -1
        if length(list_imaginary) > 0
            second_minimal_imag = abs(first(list_imaginary))
        end
        [second_minimal_abs, second_minimal_imag]
    end
    npzwrite(filename, gaps)
end

###########################################################################
############################## LQSW #######################################
###########################################################################

gens = [("ba_3", n -> multi_sink_condensation_ba(n, 3)),
        ("er_0.4", n -> random_multi_sink_orientation(n, 0.4)),
        ("ba_1", n -> multi_sink_condensation_ba(n, 1))]

for (gen_lab, gen) = gens, n = 10:10:50
    println("$gen_lab $n")
    if n > 30 && gen_lab == "er_0.4"
        println("continue")
        continue
    end
    filename = "$data_folder_name/local_gaps_$gen_lab-$n.npz"
    if isfile(filename)
        println("File exists - computation skipped")
        continue
    end
    gaps = @sync @distributed (hcat) for m = 1:500
        println("$m")
        @time g = gen(n)
        println("found")
        l = Matrix(adjacency_matrix(g, dir=:in))
        h = adjacency_matrix(Graph(g))
        s = Matrix(evolve_generator(h, local_lind(l)))
        v = eigvals(s)
        second_minimal_abs = sort(abs.(v))[2]
        list_imaginary = sort(imag.(collect(filter(x->abs(real(x)) < 1e-10 && abs(imag(x)) > 1e-10 , v))), by = x->abs(x))
        second_minimal_imag = -1
        if length(list_imaginary) > 0
            second_minimal_imag = abs(first(list_imaginary))
        end
        [second_minimal_abs, second_minimal_imag]
    end
    npzwrite(filename, gaps)
end


## #########################################################################
# ############################# NGQSW ######################################
# ##########################################################################

models = []
push!(models, (n -> barabasi_albert(n, 1), "ba_1"))
push!(models, (n -> barabasi_albert(n, 3, complete=true, is_directed=true), "ba_3_directed"))
push!(models, (n -> connected_ER(n, 0.4, is_directed=true), "er_0.4_directed"))

for n = 5:5:20, (generator, generator_label) = models
    println("$generator_label $n")
    filename = "$data_folder_name/nonmoral_gaps_$generator_label-$n.npz"
    if isfile(filename)
        println("File exists - computation skipped")
        continue
    end
    gaps = @sync @distributed (hcat) for m = 1:500
        println("$m")
        @time g = generator(n)
        a = adjacency_matrix(g, dir=:in)
        l, vset = nm_lind(a)
        h = nm_glob_ham(a)
        hrot = nm_loc_ham(vset)
        s = Matrix(evolve_generator(h, [l], hrot))
        v = eigvals(s)
        second_minimal_abs = sort(abs.(v))[2]
        list_imaginary = sort(imag.(collect(filter(x->abs(real(x)) < 1e-10 && abs(imag(x)) > 1e-10 , v))), by = x->abs(x))
        second_minimal_imag = -1
        if length(list_imaginary) > 0
            second_minimal_imag = abs(first(list_imaginary))
        end
        [second_minimal_abs, second_minimal_imag]
    end
    npzwrite(filename, gaps)
end
