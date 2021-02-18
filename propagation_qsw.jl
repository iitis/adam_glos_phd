using Distributed
addprocs(4)
@everywhere include("reproducibility.jl")
##

@everywhere using QSWalk
@everywhere using LightGraphs
@everywhere using NPZ
@everywhere using LinearAlgebra, SparseArrays
data_folder_name = "propagation_qsw"

## ######################################################################
# ######################## prob_dist ####################################
# #######################################################################

n_ex = 121
@assert isodd(n_ex)
t_ex = 15

g_ex = PathGraph(n_ex)
mid_ex = div(n_ex+1, 2)

ϱ = proj(mid_ex, n_ex)
adj = adjacency_matrix(g_ex)

s_global = evolve_generator(adj, [adj])
ϱ_global = evolve(s_global, ϱ, t_ex)
npzwrite("$data_folder_name/global.npz", real.(diag(ϱ_global)))

s_ctqw = evolve_generator(adj, [spzeros(size(adj)...)])
ϱ_ctqw = evolve(s_ctqw, ϱ, t_ex)
npzwrite("$data_folder_name/ctqw.npz", real.(diag(ϱ_ctqw)))

s_local = evolve_generator(adj, local_lind(adj))
ϱ_local = evolve(s_local, ϱ, t_ex)
npzwrite("$data_folder_name/local.npz", real.(diag(ϱ_local)))


## ######################################################################
# ####################### α calculator new ##############################
# #######################################################################
ωs = 0.:0.2:1.
max_n = 300
t_scales = 0.02:0.02:1
t_scales |> collect

##
println("Global")
αs_global = []
for t_scale = t_scales
    t = t_scale*max_n
    println("t: $t")
    n = ceil(Int, t/0.2)
    n += iseven(n)
    mid = ceil(Int, div(n+1, 2))

    g = PathGraph(n)
    h = adjacency_matrix(g)
    s = pmap(ω->evolve_generator(h, [h], ω), ωs)
    initial_state = proj(mid, n)
    println("n: $n")
    @time prob_dists = pmap(x-> diag(evolve(x, initial_state, t)), s)

    positions = -(mid-1):(mid-1)
    sec_moments = pmap(pdist -> sum(pdist .* positions.^2), prob_dists)
    push!(αs_global, log.(sec_moments))
end
npzwrite("$data_folder_name/sec_moments_global.npz", hcat(αs_global...))
##
println("Local")

αs_global = []
for t_scale = t_scales
    t = t_scale*max_n
    println("t: $t")
    n = ceil(Int, t/0.2)
    n += iseven(n)
    mid = ceil(Int, div(n+1, 2))

    g = PathGraph(n)
    h = adjacency_matrix(g)
    s = pmap(ω->evolve_generator(h, local_lind(h), ω), ωs)
    initial_state = proj(mid, n)
    println("n: $n")
    @time prob_dists = pmap(x-> diag(evolve(x, initial_state, t)), s)

    positions = -(mid-1):(mid-1)
    sec_moments = pmap(pdist -> sum(pdist .* positions.^2), prob_dists)
    push!(αs_global, log.(sec_moments))
end
npzwrite("$data_folder_name/sec_moments_local.npz", hcat(αs_global...))
