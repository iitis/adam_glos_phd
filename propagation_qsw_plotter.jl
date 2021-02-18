include("plotter_setup.jl")
data_folder_name = "propagation_qsw"

## CTQW vs GQSW vs LQSW
cla()
figure(figsize=[5,2.5])
prob_global = npzread("$data_folder_name/global.npz")
prob_local = npzread("$data_folder_name/local.npz")
prob_ctqw = npzread("$data_folder_name/ctqw.npz")
@assert length(prob_global) == length(prob_local) == length(prob_ctqw)
n = length(prob_global)
divn=div(n, 2)
positions = -divn:divn
plot(positions, prob_ctqw, "-r", label="CTQW", linewidth=1)
plot(positions, prob_global, "--k", label="GQSW")
plot(positions, prob_local, "-.b", label="LQSW")
legend(loc=1)
xlabel("position")
ylabel("measurement probability")
ylim(0., 0.055)
xlim(-divn-5, divn+5)
savefig("plots/nonmoralizing_qsw/global_vs_local_prob_dist.pdf", bbox_inches = "tight", metadata = Dict("CreationDate" => nothing))

## scaling LQSW
linestyles = [":g", "-.b", "-k"]
ω_select = [1,3,4,6]
ωs = (0.:0.2:1.)[ω_select]
max_n = 300
t_scales = 0.02:0.02:1
data = npzread("$data_folder_name/sec_moments_local.npz")[ω_select,:]
data_t = collect(t_scales .* max_n)[1:size(data,2)]

cla()
figure(figsize=[2.8, 2])
batch = 5

linetyles = ["--", ":", "-.", "-", "-", "--"]
for (i, ω) = enumerate(ωs)
    _, data_y = scaling_exponent_generator(log.(data_t), data[i,:], batch)
    plot(data_t[div(batch, 2)+1:end-div(batch, 2)], data_y, linestyle=linetyles[i], label="\$\\omega = $ω\$", linewidth=1.3)
end

legend(loc=4, fontsize=7, bbox_to_anchor=(1., 0.15))
ylim(0.9, 2.05)
#hlines([1,2], minimum(data_t), maximum(data_t), linestyles="--", linewidth=.8)

xlabel("time \$t\$")
ylabel("scaling exponent \$\\alpha \$")
savefig("plots/nonmoralizing_qsw/scaling_exponent_local.pdf", bbox_inches = "tight", metadata = Dict("CreationDate" => nothing))

## scaling LQSW
ω_select = [1,3,4,6]
ωs = (0.:0.2:1.)[ω_select]
max_n = 300
t_scales = 0.02:0.02:1
data = npzread("$data_folder_name/sec_moments_global.npz")[ω_select,:]
data_t = collect(t_scales .* max_n)[1:size(data,2)]

cla()
figure(figsize=[2.8, 2])
batch = 5

linetyles = ["--", ":", "-.", "-", "-", "--"]
for (i, ω) = enumerate(ωs)
    _, data_y = scaling_exponent_generator(log.(data_t), data[i,:], batch)
    plot(data_t[div(batch, 2)+1:end-div(batch, 2)], data_y, linestyle=linetyles[i], label="\$\\omega = $ω\$", linewidth=1.3)
end

legend(loc=4, fontsize=7, bbox_to_anchor=(1., 0.15))
ylim(0.9, 2.05)
#hlines([1,2], minimum(data_t), maximum(data_t), linestyles="--", linewidth=.8)
yticks(visible=false)

xlabel("time \$t\$")
savefig("plots/nonmoralizing_qsw/scaling_exponent_global.pdf", bbox_inches = "tight", metadata = Dict("CreationDate" => nothing))
