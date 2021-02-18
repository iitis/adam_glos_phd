using Distributed
addprocs(3)
@everywhere include("reproducibility.jl")
using Random
Random.seed!(42)
##
@everywhere using QSWalk
@everywhere using LightGraphs
@everywhere using LinearAlgebra,SparseArrays
@everywhere using NPZ
data_folder_name = "nonmoralizing_qsw"

## ######################################################################
# #################### premature localization test ######################
# #######################################################################
function gue_random(n::Int)
    @assert n >= 1
    result = zeros(ComplexF64, n, n)
    for i=1:n, j=i:n        
        result[i,j] = randn()
    end

    for i=1:n, j=(i+1):n        
        result[i,j] += 1im*randn()
        result[j,i] = conj(result[i,j])
    end
    result
end

function goe_random(n::Int)
    @assert n >= 1
    result = zeros(Float64, n, n)
    for i=1:n, j=i:n        
        result[i,j] = randn()
        result[j,i] = result[i,j]
    end
    result
end
    
repeating = 20_000

g = DiGraph(4)
add_edge!(g, 1, 2)
add_edge!(g, 2, 1)
add_edge!(g, 2, 3)
add_edge!(g, 3, 2)
add_edge!(g, 1, 3)
add_edge!(g, 3, 1)
add_edge!(g, 1, 4)

adj = adjacency_matrix(g, Float64, dir=:in)
ll, vset = QSWalk.nm_lind(adj)

data = zeros(repeating)
for m=1:repeating
    block_ham = Dict(v => gue_random(length(v)) for v=vlist(vset))
    rot_ham = nm_loc_ham(vset, block_ham)

    s = evolve_generator(zeros(size(ll)...), [ll], rot_ham)
    data[m] = sort(abs.(eigvals(s)))[2]
end
npzwrite("$data_folder_name/gue_loc_ham.npz", data)

data = zeros(repeating)
for m=1:repeating
    block_ham = Dict(v => goe_random(length(v)) for v=vlist(vset))
    rot_ham = nm_loc_ham(vset, block_ham)

    s = evolve_generator(zeros(size(ll)...), [ll], rot_ham)
    data[m] = sort(abs.(eigvals(s)))[2]
end
npzwrite("$data_folder_name/goe_loc_ham.npz", data)

## ######################################################################
# ######################### symmetry ####################################
# #######################################################################

n = 61
@assert isodd(n)
t = 100

g = PathGraph(n)
a = adjacency_matrix(g)
mid = div(n+1, 2)

l, vset  = nm_lind(a)
h_rot = nm_loc_ham(vset)

s = evolve_generator(spzeros(size(l)...), [l], h_rot)
ϱ = nm_init(vset[[mid]], vset)
ϱ_t = evolve(s, ϱ, t)
npzwrite("$data_folder_name/probdist_non_symmetric.npz", nm_measurement(ϱ_t, vset))

#

linddescription1 = Dict{Int,Matrix{Float64}}(1 => ones(1, 1), 2 => [1 1; 1 -1])
linddescription2 = Dict{Int,Matrix{Float64}}(1 => ones(1, 1), 2 => [1 1; -1 1])
lind1, vset = nm_lind(a, linddescription1)
lind2, _ = nm_lind(a, linddescription2);
h_rot = nm_loc_ham(vset)


s = evolve_generator(spzeros(size(l)...), [lind1, lind2], h_rot)
ϱ = nm_init(vset[[mid]], vset)
ϱ_t = evolve(s, ϱ, t)
npzwrite("$data_folder_name/probdist_symmetric.npz", nm_measurement(ϱ_t, vset))

## ######################################################################
# ######################### propagation ##################################
# #######################################################################

ωs = 0.5:0.1:1.
max_n = 400
t_scales = 0.1:0.1:10
scaling = 1/3
#
αs_global = []
for t_scale = t_scales
    t = t_scale*max_n
    n = max(max_n, ceil(Int, t*scaling))
    n += iseven(n)
    mid = ceil(Int, div(n+1, 2))

    g = PathGraph(n)
    a = adjacency_matrix(g)
    linddescription1 = Dict{Int,Matrix{Float64}}(1 => ones(1, 1), 2 => [1 1; 1 -1])
    linddescription2 = Dict{Int,Matrix{Float64}}(1 => ones(1, 1), 2 => [1 1; -1 1])
    lind1, vset = nm_lind(a, linddescription1)
    lind2, _ = nm_lind(a, linddescription2)
    h_rot = nm_loc_ham(vset)
    h = nm_glob_ham(a)

    s = pmap(ω->evolve_generator(h, [lind1, lind2], h_rot, ω), ωs)
    println(typeof(s))
    initial_state = nm_init(vset[[mid]], vset)
    @time prob_dists = pmap(x-> nm_measurement(evolve(x, initial_state, t), vset), s)
    
    positions = -(mid-1):(mid-1)
    sec_moments = pmap(pdist -> sum(pdist .* positions.^2), prob_dists)
    push!(αs_global, log.(sec_moments))
    npzwrite("$data_folder_name/scaling_data.npz", hcat(αs_global...))
end
