include("plotter_setup.jl")

using LambertW
using Statistics

function theoretical_p0(p0)
    x = (1-p0)/exp(1)/p0
    lambertw(x, 0)/lambertw(x, -1)
end

## p0 theoretical only

data_x = (1.:.1:10000)
data_p = theoretical_p0.(data_x)

cla()
figure(figsize=[4, 2.])
plot(data_x, data_p, "-k")
ylim(0, 1)
xlim(.9, 11000)
xscale("log")
xlabel(L"p_0")
ylabel("succ. prob. lower bound")
yticks([0, 0.5, 1.])
savefig("../hiding_search/p0_theoretical.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

## p0 real scenarios
p0s = .5:.5:2.
p(p0, n) = p0*log(n)/n
repeating = 40

data_n = 100:100:2000

fig, axes = subplots(1, length(p0s), figsize=[6, 2.])
println(p0s)
cla()
for (p0, ax) = zip(p0s, axes)
    data = npzread("hiding_search/sparse_ER_probs_and_bounds_$p0.npz")
    data_real_p = zeros(length(data_n))
    data_real_p_up = zeros(length(data_n))
    data_real_p_down = zeros(length(data_n))
    data_real_bound = zeros(length(data_n))
    data_real_bound_up = zeros(length(data_n))
    data_real_bound_down = zeros(length(data_n))
    for m =1:length(data_n)
        data_b = data[m,:,1]
        data_p = data[m,:,2]

        data_real_bound[m] = mean(data_b)
        data_real_bound_up[m] = maximum(data_b)
        data_real_bound_down[m] = minimum(data_b)
        
        data_real_p[m] = mean(data_p)
        data_real_p_up[m] = maximum(data_p)
        data_real_p_down[m] = minimum(data_p)
    end

    if p0 > 1
        ax.plot(data_n, data_real_bound, "-b", markersize=.8, linewidth=.8)
        ax.fill_between(data_n, data_real_bound_up, data_real_bound_down, color="b", alpha=.2)
    end
    ax.plot(data_n, data_real_p, ":r", markersize=.8, linewidth=.8)
    ax.fill_between(data_n, data_real_p_up, data_real_p_down, color="r", alpha=.2)
    
    if p0 > 1
        ax.plot([minimum(data_n), maximum(data_n)], fill(theoretical_p0(p0), 2), "--k", linewidth=.8)
    end
    ax.set_title(latexstring("p_0=$p0"))
    ax.set(xlabel=L"n", ylim=[0,1], yticks=0:0.5:1.) 
    ax.label_outer()   
end
axes[1].set(ylabel="succ. prob.")
savefig("../hiding_search/real_statistics_p0.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))