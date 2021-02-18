include("plotter_setup.jl")
using Statistics
using LinearAlgebra

## convergence in prob NGQSW

models = ["ba_1", "ba_3", "er_0.4"]
n = 12

mod_labs = ["{\\mathcal G}^{\\rm BA}_{12}(1)",
            "{\\mathcal G}^{\\rm BA}_{12}(3)",
            "{\\mathcal G}^{\\rm ER}_{12}(0.4)"]

t_step = 100
data_t = 0:100:10_000

cla()

function t_monotonic(data_p_diff::Vector{<:Real}, data_t)
    diffs = [j-i for (i,j) = zip(data_p_diff, data_p_diff[2:end])]
    res = findlast(x->x > 0., diffs) # because we are interested in the right which is violated
    if res == nothing 
        res = 0
    end
    if res > 2*length(data_p_diff)/3 
        @show maximum(abs.(data_p_diff[end-30:end]))
    end
    data_t[res+1]
end

for gtype = ["_directed", ""]
    cla()
    figure(figsize=[2,1.8])
    for (model, mod_lab) = zip(models, mod_labs)
        data_raw = npzread("convergence_qsw/nonmoral_probs_$(model)$gtype-$n.npz")
        data_y = Float64[]
        data_probs = mapslices(x -> x./sum(x), data_raw, dims=1)
        for m=1:size(data_raw,3)
            p_last = data_probs[:,end,m]
            data_p_diff = mapslices(x->norm(x - p_last), data_probs[:,:,m], dims=1)[:]
            push!(data_y, t_monotonic(data_p_diff, data_t))
        end
        label = latexstring(gtype=="_directed" ? "\\vec" : "", mod_lab)
        hist(data_y, bins=-250:500:10_250 , label=label, alpha=.5, histtype="stepfilled")
        
    end
    

    xlabel(L"t_{\rm min}")
    xlim(-400, 10_400)
    ylim(0, 500)
    if gtype != "_directed" 
        tick_params(labelleft=false)
    else
        ylabel("frequency")
    end
    legend(loc=1)

    savefig("plots/convergence_qsw/nonmoral_statistics_probs$gtype.pdf",bbox_inches="tight", metadata = Dict("CreationDate" => nothing))
end

## special graph

times = 0:10:1000
n = 7
result1 = npzread("convergence_qsw/special_nmgqsw_1.npz")
result2 = npzread("convergence_qsw/special_nmgqsw_2.npz") 

cla()

fig = figure(figsize=[3,1.8])


plot(times, result1[5,:], "k-", label=L"\varrho=\varrho'")
plot(times, result2[5,:], "b--", label=L"\varrho=\varrho''")

ylabel(L"p(t, \varrho)(5)")
xlabel("evolution time")
legend(loc=4)
savefig("plots/convergence_qsw/special_nmgqsw.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

## path structure observance
ωs = 0.5:0.05:1.
t_inf = 10_000
repeating = 50
n = 15

cla()

data_l = npzread("convergence_qsw/local_structure_observance_path_15.npz")
data_g = npzread("convergence_qsw/global_structure_observance_path_15.npz")
data_n = npzread("convergence_qsw/nonmoral_structure_observance_path_15.npz")

fig, axes = subplots(2, 1, figsize=[4,3])

axes[1,1].plot(ωs, [data_l[ωind, 1] for ωind=1:length(ωs)], "kx-", label="LQSW")
axes[1,1].plot(ωs, [data_g[ωind, 1] for ωind=1:length(ωs)], "b1--", label="GQSW")
axes[1,1].plot(ωs, [data_n[ωind, 1] for ωind=1:length(ωs)], "r2:", label="NQSW")

axes[1,1].set_ylim(-0.05, 1.05)
axes[1,1].set_xlim(.45, 1.05)
axes[1,1].set_xlim(.45, 1.05)
axes[1,1].set(ylabel=L"p_s")
axes[1,1].get_xaxis().set_ticklabels([])

axes[2,1].plot(ωs, [sum(data_l[ωind, :].*(0:(n-1)).^2) for ωind=1:length(ωs)], "kx-", label="LQSW")
axes[2,1].plot(ωs, [sum(data_g[ωind, :].*(0:(n-1)).^2) for ωind=1:length(ωs)], "b1--", label="GQSW")
axes[2,1].plot(ωs, [sum(data_n[ωind, :].*(0:(n-1)).^2) for ωind=1:length(ωs)], "r2:", label="NMQSW")

axes[2,1].set_xlim(.45, 1.05)
axes[2,1].legend(loc=1)
axes[2,1].set(ylabel=L"\mu_s", xlabel=L"\omega")
savefig("plots/convergence_qsw/structure_observance_path.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))

## ba structure observance
cla()


fig, axes = subplots(2, 2, figsize=[4.8,3])

for m0=[1,2]
    data_l = npzread("convergence_qsw/local_structure_observance_$n-$m0.npz")
    data_g = npzread("convergence_qsw/global_structure_observance_$n-$m0.npz")
    data_n = npzread("convergence_qsw/nonmoral_structure_observance_$n-$m0.npz")   
    labels = ["LQSW", "GQSW", "NGQSW"]
    colors = ["r", "b", "g"]
    linestyles = [":", "--", "-"]

    for (data, label, c, l) = zip([data_l,data_g,data_n], labels, colors, linestyles)
        data_p = zeros(repeating, length(ωs))
        data_mu = zeros(repeating, length(ωs))
        for m=1:repeating, i=1:length(ωs)
            data[m,i,:] /= sum(data[m,i,:])
            data_p[m,i] = sum(data[m,i,1:m0])
            data_mu[m,i] = sum(data[m,i,:] .* (data[m,end,:]).^2)
        end
        println(size(data_p))
        data_p_mean = mapslices(mean, data_p, dims=1)[:]
        data_p_top = mapslices(maximum, data_p, dims=1)[:]
        data_p_bottom  = mapslices(minimum, data_p, dims=1)[:]
        if m0 == 2 && c == "g"
            println(sort(data_p[:,end]))
            println("$(length(filter(x-> x<.99, data_p[:,end]))) / $(length(data_p[:,end]))")
            println("$(length(filter(x-> x<.99, data_p[:,end]))) / $(length(data_p[:,end]))")
        end

        data_mu_mean = mapslices(mean, data_mu, dims=1)[:]
        data_mu_top = mapslices(maximum, data_mu, dims=1)[:]
        data_mu_bottom  = mapslices(minimum, data_mu, dims=1)[:]
        
        if m0 == 2
            axes[1,m0].plot(ωs, data_p_mean, "$c$l", label=label)
        else
            axes[1,m0].plot(ωs, data_p_mean, "$c$l")
        end
        axes[1,m0].fill_between(ωs, data_p_bottom, data_p_top, color=c,alpha=0.3)

        axes[2,m0].plot(ωs, data_mu_mean, "$c$l")
        axes[2,m0].fill_between(ωs, data_mu_bottom, data_mu_top, color=c,alpha=0.3)
        
        

        axes[1,m0].set_ylim(-0.05, 1.05)
        axes[1,m0].set_xticks([0.5, .75, 1.])
        axes[2,m0].set_xticks([0.5, .75, 1.])
        axes[1,m0].set_title(latexstring("m_0 = $m0"))
        
        axes[1,m0].get_xaxis().set_ticklabels([])
        
    end
end
axes[1,2].get_yaxis().set_ticklabels([])
axes[2,2].get_yaxis().set_ticklabels([])
axes[1,1].set(ylabel=L"p_s")
axes[2,1].set(ylabel=L"\mu_s")
axes[2,1].set(xlabel=L"\omega")
axes[2,2].set(xlabel=L"\omega")
axes[1,2].legend(loc=1,bbox_to_anchor=[1.7,1.05])
    
savefig("plots/convergence_qsw/structure_observance_ba.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))    

## global

ns = 10:10:50
models = ["ba_1", "ba_3", "er_0.4"]



for model = models
    println(model)
    cla()
    figure(figsize=[1.5,1.6])
    data_relaxing = Int[]
    data_converging_nonrelaxing = Int[]
    data_maybeperiodic = Int[]
    for n = ns
        data = npzread("convergence_qsw/global_gaps_$model-$n.npz")
        N = length(data[1,:])

        real_data = collect(filter(x -> x> 1e-10, sort(data[1,:])))
        push!(data_relaxing, length(real_data))
        
        imag_data = sort(collect(filter(x->x != -1, data[2,:])))

        push!(data_converging_nonrelaxing, N - length(imag_data)- data_relaxing[end])
        push!(data_maybeperiodic, N - data_converging_nonrelaxing[end]- data_relaxing[end])
    end
    println(data_relaxing)
    println(data_converging_nonrelaxing)
    println(data_maybeperiodic)
    bar(collect(ns), data_maybeperiodic, bottom=data_converging_nonrelaxing .+ data_relaxing, color="b", label="all", width=4, edgecolor="black",  hatch="\\\\\\\\", linewidth=.2)
    bar(collect(ns), data_converging_nonrelaxing, bottom=data_relaxing, color="k", edgecolor="black", label="convergent", width=4, linewidth=.2)
    bar(collect(ns), data_relaxing, color="r", label="relaxing", width=4, edgecolor="black", hatch="////",  linewidth=.2)
    xlim(5,55)
    ylim(0,525)
    xlabel("graph order \$n\$")
    if model == models[end]
        legend(loc=1,bbox_to_anchor=(1.85, 1.02), fontsize=7)
    end
    if model != models[1]
        yticks(visible=false)
    end

    savefig("plots/convergence_qsw/global_statistics_convergence_$model.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))
end

## local
ns = 10:10:50
models = ["ba_1", "ba_3", "er_0.4"]

for model = models
    println(model)
    cla()
    figure(figsize=[1.5,1.6])
    #title(generate_title(model))
    data_relaxing = Int[]
    data_converging_nonrelaxing = Int[]
    data_maybeperiodic = Int[]
    for n = ns
        if !isfile("convergence_qsw/local_gaps_$model-$n.npz")
            push!(data_relaxing, 0)
            push!(data_converging_nonrelaxing, 0)
            push!(data_maybeperiodic, 0)
            continue
        end

        data = npzread("convergence_qsw/local_gaps_$model-$n.npz")
        N = length(data[1,:])

        real_data = collect(filter(x -> x> 1e-10, sort(data[1,:])))
        push!(data_relaxing, length(real_data))

        imag_data = sort(collect(filter(x->x != -1, data[2,:])))

        push!(data_converging_nonrelaxing, N - length(imag_data)- data_relaxing[end])
        push!(data_maybeperiodic, N - data_converging_nonrelaxing[end]- data_relaxing[end])
    end
    println(data_relaxing)
    println(data_converging_nonrelaxing)
    println(data_maybeperiodic)
    @show "test"
    bar(collect(ns), data_maybeperiodic, bottom=data_converging_nonrelaxing .+ data_relaxing, color="b", label="all", width=4, edgecolor="black",  hatch="\\\\\\\\", linewidth=.2)
    bar(collect(ns), data_converging_nonrelaxing, bottom=data_relaxing, color="k", label="convergent", edgecolor="black", width=4, linewidth=.2)
    bar(collect(ns), data_relaxing, color="r", label="relaxing", width=4, edgecolor="black", hatch="////",  linewidth=.2)
    xlim(5,55)
    ylim(0,525)
    xlabel("graph order \$n\$")
    if model == models[end]
        legend(loc=1,bbox_to_anchor=(1.85, 1.02), fontsize=7)
    end
    if model != models[1]
        yticks(visible=false)
    end

    savefig("plots/convergence_qsw/local_statistics_convergence_$model.pdf", bbox_inches="tight", metadata = Dict("CreationDate" => nothing))
end


## ngqsw 

ns = 5:5:20
models = ["ba_1", "ba_3_directed", "er_0.4_directed"]

for model = models
    println(model)
    cla()
    figure(figsize=[1.5,1.6])
    #title(generate_title(model))
    data_relaxing = Int[]
    data_converging_nonrelaxing = Int[]
    data_maybeperiodic = Int[]
    for n = ns
        if !isfile("convergence_qsw/nonmoral_gaps_$model-$n.npz")
            push!(data_relaxing, 0)
            push!(data_converging_nonrelaxing, 0)
            push!(data_maybeperiodic, 0)
            continue
        end

        data = npzread("convergence_qsw/nonmoral_gaps_$model-$n.npz")

        N = length(data[1,:])

        real_data = collect(filter(x -> x> 1e-10, sort(data[1,:])))
        push!(data_relaxing, length(real_data))

        imag_data = sort(collect(filter(x->x != -1, data[2,:])))

        push!(data_converging_nonrelaxing, N - length(imag_data)- data_relaxing[end])
        push!(data_maybeperiodic, N - data_converging_nonrelaxing[end]- data_relaxing[end])
    end
    println(data_relaxing)
    println(data_converging_nonrelaxing)
    println(data_maybeperiodic)
    bar(collect(ns), data_maybeperiodic, bottom=data_converging_nonrelaxing .+ data_relaxing, color="b", label="all", width=2, edgecolor="black",  hatch="\\\\\\\\", linewidth=.2)
    bar(collect(ns), data_converging_nonrelaxing, bottom=data_relaxing, color="k", label="convergent", width=2, linewidth=.2)
    bar(collect(ns), data_relaxing, color="r", label="relaxing", width=2, edgecolor="black", hatch="////",  linewidth=.2)
    xlim(3,22)
    ylim(0,525)
    xlabel("graph order \$n\$")
    if model == models[end]
        legend(loc=1,bbox_to_anchor=(1.85, 1.02), fontsize=7)
    end
    if model != models[1]
        yticks(visible=false)
    end

    savefig("plots/convergence_qsw/nonmoral_statistics_convergence_$model.pdf",bbox_inches="tight", metadata = Dict("CreationDate" => nothing))
end
## special graph




