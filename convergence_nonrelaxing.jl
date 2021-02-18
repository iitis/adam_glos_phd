using Distributed 
addprocs(8)
@everywhere include("reproducibility.jl")
include("reproducibility.jl")
@everywhere using LightGraphs
@everywhere using LinearAlgebra
@everywhere using QSWalk
@everywhere using SparseArrays
@everywhere using NPZ

data_folder_name = "convergence_qsw"

##


a = [0 0 0 0 1 1 0; 0 0 0 0 1 0 0; 0 0 0 0 1 1 0; 0 0 0 0 1 1 0; 1 1 1 1 0 1 0; 1 0 1 1 1 0 1; 0 0 0 0 0 1 0]
l, vset = nm_lind(a)
h = nm_glob_ham(a)
hrot = nm_loc_ham(vset)
s = sparse(evolve_generator(h, [l], hrot))

init1 = nm_init(vset[[1]], vset)
init2 = nm_init(vlist(vset), vset)
times = 0:10:1000

println("result 1")
result1 = hcat(pmap(t->nm_measurement(evolve(s, init1, t), vset), times)...)
npzwrite("$data_folder_name/special_nmgqsw_1.npz", result1)
println("result 2")
result2 = hcat(pmap(t->nm_measurement(evolve(s, init2, t), vset), times)...)
npzwrite("$data_folder_name/special_nmgqsw_2.npz", result2)




