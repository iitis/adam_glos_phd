include("plotter_setup.jl")

using LightGraphs
using Expokit

##

n = 101
t = 22

g = PathGraph(n)
a = adjacency_matrix(g)
l = laplacian_matrix(g)

init_state = zeros(Float64, n)
init_state[div((n+1),2)] = 1.
##

classical_dist = expmv(t, -l, init_state)
@assert sum(classical_dist) ≈ 1.
quantum_dist = abs2.(expmv(t, 1im*a, init_state))
@assert sum(quantum_dist) ≈ 1.

##
using PyPlot
rc("text", usetex=true)
rc("font", family="serif")

plot_style= (markersize=1)

f = figure(figsize=(2.5,2))
bound = div(n, 2)
plot(-bound:bound, classical_dist,"ko-", markersize=2,linewidth=.4)
ylim(-0.005,0.085)
ylabel("probability")
xlabel("position")
savefig("../introduction/classical.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

f = figure(figsize=(2.5,2))
plot(-bound:bound, quantum_dist,"ko-", markersize=2,linewidth=.4)
ylim(-0.005,0.085)
yticks(visible=false)
xlabel("position")
savefig("../introduction/quantum.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))


