include("plotter_setup.jl")
using LinearAlgebra

# visualization cl
cla()
fig, axes = subplots(1, 3, figsize=[6.,1.8])

for (ax, a, b) = zip(axes, [.01, .01, .01], [.2, .5, .99])
    @assert a < 3*b
    @assert a + b <= 1

    l1 = ax.plot([0, 1], [(1+b)/2, 1/2], "-k", label="norm. Laplacian")
    l2 = ax.plot([0, 1], [1/2+b, 1/2], "--r", label="adjacency matrix")
    l3 = ax.plot([0, 1], [(1+b), 1], "b", label="classical", linestyle="dashdot")
    setp(ax, xlabel=L"degree exponent $a+ \frac{i}{n}b$")
    ax.hlines([0.5, 1.], -0.05, 1.05, linestyle=":", color="k", linewidth=1)
    ax.text(.55, 1.75, latexstring("b = $b"))
end

setp(axes, yticks=0:0.5:2, yticklabels=fill("", 5), xticks=[0,1], xticklabels=[L"a", L"a+b"])
setp(axes[1], ylabel="time exponent", yticklabels=0:0.5:2)
setp(axes, ylim=[-0.0,2.0], xlim=[-0.0,1.0])

handles, labels = axes[1].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="upper center", ncol=3,framealpha=0)
leg.get_frame().set_linewidth(0.0)
savefig("../complex_search/cl_visualization.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

## BA 

m0 = 3
nodes = [1, :n]
repeating = 20 #may be smaller than maximal
data_n = 100:100:5_000
matrices = ["norm. laplacian", "laplacian", "adjacency matrix"]

for node=nodes   
    cla()
    fig, axs = subplots(2, 3, figsize=(6,3.))

    for (i, matrix) = enumerate(matrices), m = 1:repeating
        color = 3/4* (1-m/repeating) .* (1,1,1)
        data_raw = npzread("complex_search/BA-$m0-$node-$(matrix[1:3])-$m.npz")
        axs[1,i].plot(data_n, data_raw[:,end] , "-k", linewidth=.1, color=color)
        axs[2,i].loglog(data_n, data_raw[:,end-1] ./ data_raw[:,end], "-", linewidth=.1, color=color)
        axs[1,i].set_title(replace(matrix, "laplacian" => "Laplacian"))
    end

    axs[1, 1].set(ylabel=L"succ. prob. $p(T)$")
    axs[2, 1].set(ylabel=L"exp. time $T/p(T)$")

    axs[1,2].tick_params(labelleft=false)
    axs[1,3].tick_params(labelleft=false)
    axs[2,2].tick_params(labelleft=false)
    axs[2,3].tick_params(labelleft=false)

    for i=1:3
        axs[1,i].set_xscale("log")
        axs[1,i].tick_params(labelbottom=false)
        axs[1,i].set_xlim(95,6_000)
        axs[2,i].set_xlim(95,6_000)
        axs[2,i].set(xlabel=L"graph order $n$")
        axs[2,i].tick_params(which="minor", left=true, labelleft=false)
        axs[2,i].set_yticks(vcat([ (2*k):(2*k):(9*k) for k=10 .^ (0:4)]...), minor=true)
        axs[2,i].set_yticks(10 .^ (0:4))
        axs[1,i].set_yticks(0:0.2:1.)
        axs[1,i].set_ylim(-0.05,1.05)
        axs[2,i].set_ylim(2, 15_000)
    end    

    savefig("../complex_search/ba_$m0-$node.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))
end

repeating = 200 # consider all!
for node=nodes
    cla()
    fig, axs = subplots(2, 3, figsize=(6,3.))
    for (i,matrix) = enumerate(matrices)
        data_alpha = zeros(repeating)
        data_reg_qual = zeros(repeating)

        for m = 1:repeating
            data_raw = npzread("complex_search/BA-$m0-$node-$(matrix[1:3])-$m.npz")
            data_y = log.(data_raw[:,end-1] ./ data_raw[:,end])
            beta, alpha = linreg(log.(data_n), data_y)
            data_alpha[m] = alpha
            data_reg_qual[m] = norm(data_y .- alpha*log.(data_n) .- beta)
        end
        @show maximum(data_reg_qual)
        axs[1,i].hist(data_alpha, bins=-0.0:.05:2.0, color="k", linewidth=.1)
        axs[2,i].plot(data_alpha, data_reg_qual, "kx")        
        axs[1,i].set_title(replace(matrix, "laplacian" => "Laplacian"))
    end

    axs[1,2].tick_params(labelleft=false)
    axs[1,3].tick_params(labelleft=false)
    axs[2,2].tick_params(labelleft=false)
    axs[2,3].tick_params(labelleft=false)
    
    axs[1,1].set_ylabel("frequency")
    axs[2,1].set_ylabel("regression quality")
    for i=1:3
        axs[1,i].tick_params(labelbottom=false)
        axs[2,i].set(xlabel=L"exponent $\alpha$")
        axs[1,i].tick_params(which="minor", left=true, labelleft=false)
        axs[1,i].set_xticks(.25:.5:1.75, minor=true)
        axs[1,i].set_xticks(0.:.5:2.)
        axs[2,i].tick_params(which="minor", left=true, labelleft=false)
        axs[2,i].set_xticks(.25:.5:1.75, minor=true)
        axs[2,i].set_xticks(0.:.5:2.)

        
        axs[1,i].set_xlim(-0.1,2.1)
        axs[2,i].set_xlim(-0.1,2.1)
        axs[1,i].set_ylim(0, 130) #TODO fill
        axs[2,i].set_ylim(-0.2, 10.2) #TODO fill
    end    

    savefig("../complex_search/ba_$m0-$node-exponent.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))
end
