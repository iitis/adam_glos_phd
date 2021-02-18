include("plotter_setup.jl")
using LsqFit
data_folder_name = "nonmoralizing_qsw"


data_goe = log10.(npzread("$data_folder_name/goe_loc_ham.npz")[:,1])
data_gue = log10.(npzread("$data_folder_name/gue_loc_ham.npz")[:,1])
n = length(data_goe)
@assert length(data_gue) == n

cla()
figure(figsize=[4, 2.4])

hist(data_goe, label="GOE", color="g", bins=70, alpha=.4, ec="k", histtype="stepfilled")
hist(data_gue, label="GUE", color="r", bins=70, alpha=.7, ec="k", histtype="stepfilled")
ylabel("frequency")

ylims = ylim()
vlines(minimum(data_goe), ylims..., color="b", linestyle="-", linewidth=.5)
vlines(minimum(data_gue), ylims..., color="r", linestyle="--", linewidth=.5)
ylim(ylims...)
xlabel(L"\log_{10} (|\lambda|)")
leg = legend(loc=2)

savefig("../nonmoralizing_qsw/loc_ham_analysis.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

##

data_nonsym = npzread("$data_folder_name/probdist_non_symmetric.npz")
data_sym = npzread("$data_folder_name/probdist_symmetric.npz")
n = length(data_sym)
#

positions = -(div(n, 2)):(div(n, 2))
markersize = 3
mew = .5
linewidth = .3

cla()
figure(figsize=[2.8, 2])

plot(positions, data_nonsym, "-+r", label="NGQSW", linewidth=linewidth, markersize=markersize+1, mew=mew)
plot(positions, reverse(data_nonsym), "-xk", label="NGQSW rev.", linewidth=linewidth, markersize=markersize, mew=mew)
vlines(0, 0, .06, linestyles="--", colors="k",linewidth=.5)

legend(loc=1, fontsize=6)
ylim(-.005, 0.065)
xlabel("position")
ylabel("measurement probability")
savefig("../nonmoralizing_qsw/nonsymmetric.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

cla()
figure(figsize=[2.8, 2])

plot(positions, data_sym, "-+r", label="NGQSW", linewidth=linewidth, markersize=markersize+1, mew=mew)
plot(positions, reverse(data_sym), "-xk", label="NGQSW rev.", linewidth=linewidth, markersize=markersize, mew=mew)
vlines(0, 0, .06, linestyles="--", colors="k",linewidth=.5)

legend(loc=1, fontsize=6)
ylim(-.005, 0.065)
yticks(visible=false)
xlabel("position")

savefig("../nonmoralizing_qsw/symmetric.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

## scaling new
ω_select = [1,2,4,6]
ωs = (0.5:0.1:1.)[ω_select]
max_n = 300
t_scales = 0.1:0.1:6
t_scales_cont = 0.1:0.1:6
batch = 5

data = npzread("$data_folder_name/scaling_data.npz")[ω_select,:]

data_t = collect(t_scales .* max_n)[1:size(data,2)]
data_t_cont = collect(t_scales_cont .* max_n)
println(data_t[[1,end]])

cla()
figure(figsize=[5, 2.4])
b_half = div(5, 2)

#linetyles = ["--", ":", "-.", "-"]
style = ["+", ".", "d", "x"]
marksizes = [5, 4.5, 3, 4]

for (i, ω) = enumerate(ωs)
    _, data_y = scaling_exponent_generator(log.(data_t), data[i,:], batch)
    
    plot(data_t[b_half+1:end-b_half], data_y, "$(style[i])", label=latexstring("\\omega = $ω"), markersize=marksizes[i])
    if i < 4
        @. model(x, p) = p[1] - p[2]/(x-p[3])^p[4]
        lb = [1.5, 10. , -Inf, 0.]
        ub = [2.5, Inf, 20, 1]
        p0 = [2., 15, 1, 1]
        fit = curve_fit(model, data_t[b_half+20:end-b_half], data_y[20:end], p0, lower=lb, upper=ub)
        @show length(data_y[20:end])
        c1, c2, c3, α = fit.param
        @show ω, fit.param
        @show stderror(fit)

        plot(data_t_cont, c1 .- c2 ./ (data_t_cont .- c3) .^ α, "--k", linewidth=.8)
        println()
    end
end

legend(loc=4, fontsize=7, bbox_to_anchor=(1., 0.15))
hlines([1,2], 0, maximum(data_t_cont), linestyles="-", linewidth=.6)

xlabel("time \$t\$")
ylabel("scaling exponent \$\\alpha_t \$")
savefig("../nonmoralizing_qsw/scaling.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

