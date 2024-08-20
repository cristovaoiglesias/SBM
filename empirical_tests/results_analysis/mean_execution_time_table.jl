
using CSV, DataFrames
using Printf
using LinearAlgebra, DifferentialEquations
# using Plots; gr();
using Plots,Noise
using StatsPlots
using Dates
using Distributions
using Turing, Measures
# using  OptimizationOptimJL, OptimizationFlux
# using  ModelingToolkit , Optimization
# using  Optimization, OptimizationBBO
# using Symbolics
using LaTeXStrings
using MCMCChains

using Random
Random.seed!(1);



using Statistics
using StatsBase



FULL_PATH="" #e.g. "/Users/Thesis"



## PUB for BBF1 and BBF2 with NARROW surrogate distribution of exp2
# load all results of exp2 related to PUB-BBF1
directory =  FULL_PATH*"/SBM/empirical_tests/task1/step3/set100"
all_files = readdir(directory)
only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)

for  sizedt in [1,2,3,4,5,10,25,50,75,104]
    all_chains_bf=[]
    all_times=[]
    global casd=1
    for i in txt_files
        if occursin("chain_bbf1_$(sizedt)_",i )
            timeRun=parse(Float64, split(i, "_time_")[2][1:9])
            # println(casd ,"| filename: ",i," | time: ",timeRun)
            if timeRun < 100 #removed the outliers
                append!(all_times,timeRun)
            end
            global casd=casd+1
            ch = CSV.read(directory*"/"*i, DataFrame)
            append!(all_chains_bf,[ch])
        end
    end
    println("PUB-BBF1 (Narrow surrugate dist size $(sizedt)) - Mean of the time of all runs: ",round(mean(all_times), digits=1),"±",round(std(all_times), digits=1))
end



for  sizedt in [1,2,3,4,5,10,25,50,75,104]
    all_chains_bf=[]
    all_times=[]
    global casd=1
    for i in txt_files
        if occursin("chain_bbf2_$(sizedt)_",i )
            timeRun=parse(Float64, split(i, "_time_")[2][1:9])
            # println(casd ,"| filename: ",i," | time: ",timeRun)
            if timeRun < 100 #removed the outliers
                append!(all_times,timeRun)
            end
            global casd=casd+1
            ch = CSV.read(directory*"/"*i, DataFrame)
            append!(all_chains_bf,[ch])
        end
    end
    println("PUB-BBF2 (Narrow surrugate dist size $(sizedt)) - Mean of the time of all runs: ",round(mean(all_times), digits=1),"±",round(std(all_times), digits=1))
end


















## PUB for BBF1 and BBF2 with wide surrogate distribution of exp3
# load all results of exp3 related to PUB-BBF1
directory =  FULL_PATH*"/SBM/empirical_tests/task1/step4/set100"
all_files = readdir(directory)
only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)


all_chains_bf=[]
all_times=[]
casd=1
for i in txt_files
    if occursin("chain_bbf1_",i )
        timeRun=parse(Float64, split(i, "_time_")[2][1:9])
        println(casd ,"| filename: ",i," | time: ",timeRun)
        if timeRun < 77 #removed the outliers
            append!(all_times,timeRun)
        end
        global casd=casd+1
        ch = CSV.read(directory*"/"*i, DataFrame)
        append!(all_chains_bf,[ch])
    end
end

println("PUB-BBF1 (wide surrugate dist) - Mean of the time of all runs: ",round(mean(all_times), digits=1),"±",round(std(all_times), digits=1))

# load all results of exp3 related to PUB-BBF2
all_chains_bf=[]
all_times=[]
casd=1
for i in txt_files
    if occursin("chain_bbf2_",i )
        timeRun=parse(Float64, split(i, "_time_")[2][1:4])
        println(casd ,"| filename: ",i," | time: ",timeRun)
        if timeRun < 88 #!= 88.209868 #removed the outliers
            append!(all_times,timeRun)
        end
        global casd=casd+1
        ch = CSV.read(directory*"/"*i, DataFrame)
        append!(all_chains_bf,[ch])
    end
end

println("PUB-BBF2 (wide surrugate dist) - Mean of the time of all runs: ",round(mean(all_times), digits=1),"±",round(std(all_times), digits=1))

histogram(all_times)


corbbf1=:purple2
corbbf1std104=:royalblue1
va=0.05
## Classical Bayesian Framework - load all results of exp4 related to RQ4
directory =  FULL_PATH*"/SBM/empirical_tests/task1/step5/set100"
all_files = readdir(directory)
only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)
casd=1

all_chains_bf=[]
all_times=[]
plotacf=plot(layout=(6,1),label=false);
for i in txt_files
    if occursin("classical_bf_",i )
        timeRun=parse(Float64, split(i, "_time_")[2][1:9])
        println(casd ,"| filename: ",i," | time: ",timeRun)
        if timeRun < 100 #runs above of 100 sec were removed as outliers
            append!(all_times,timeRun)
            ch = CSV.read(directory*"/"*i, DataFrame)
            lws=2.5
            gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=8);
            plot!(plotacf,autocor(ch[:,3]), ylabel = L"ACF~~\mu_{max}", lw=lws, alpha = va,subplot=1,label=false,c=corbbf1)
            plot!(plotacf,autocor(ch[:,4]), ylabel = L"ACF~~K_{lysis}", lw=lws,alpha = va,subplot=2,label=false,c=corbbf1)
            plot!(plotacf,autocor(ch[:,5]), ylabel = L"ACF~~Kd_{gln}", lw=lws,alpha = va,subplot=3,label=false,c=corbbf1)
            plot!(plotacf,autocor(ch[:,6]), ylabel = L"ACF~~Y_{lac,glc}", lw=lws,alpha = va,subplot=4,label=false,c=corbbf1)
            plot!(plotacf,autocor(ch[:,7]), ylabel = L"ACF~~Y_{amm,gln}",lw=lws, alpha = va,subplot=5,label=false,c=corbbf1)
            plot!(plotacf,autocor(ch[:,8]), ylabel = L"ACF~~\lambda", lw=lws,alpha = va,xlabel="Lag", subplot=6,label=false,c=corbbf1)
            plot!(left_margin=4mm,bottom_margin=0mm,  size=(400,800), grid=false)
            # annotate!(3., .15,ylabel="ACF", text("BAT-WP using training set A1-Reg1", :left, 10, 8))
            # display(plotacf)
        end
        global casd=casd+1
        ch = CSV.read(directory*"/"*i, DataFrame)
        append!(all_chains_bf,[ch])
    end
end
display(plotacf)
println("Classical Bayesian Framework - Mean of the time of all runs (except outliers): ",round(mean(all_times), digits=1),"±",round(std(all_times), digits=1))




































#
#
# ##
# #ACF plot between Classical Bayesian Inference and SBM
#
#
# corbbf1=:purple2
# corbbf1std104=:navyblue
#
# va=0.075
#
#
# # # PUB for BBF1 with NARROW surrogate distribution of exp2
# # load all results of exp2 related to PUB-BBF1
# directory =  FULL_PATH*"/PUB/empirical_tests/RQ2/set100"
# all_files = readdir(directory)
# only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
# txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)
# plotacf=plot(layout=(6,1),label=false);
#
# for  sizedt in [104]
#     all_chains_bf=[]
#     all_times=[]
#     global casd=1
#     for i in txt_files
#         if occursin("chain_bbf1_$(sizedt)_",i )
#             timeRun=parse(Float64, split(i, "_time_")[2][1:9])
#             # println(casd ,"| filename: ",i," | time: ",timeRun)
#             if timeRun < 100 #removed the outliers
#                 append!(all_times,timeRun)
#                 ch = CSV.read(directory*"/"*i, DataFrame)
#                 lws=2.5
#                 gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=8);
#                 plot!(plotacf,autocor(ch[:,3]), ylabel = L"ACF~~\mu_{max}", lw=lws, alpha = va,subplot=1,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,4]), ylabel = L"ACF~~K_{lysis}", lw=lws,alpha = va,subplot=2,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,5]), ylabel = L"ACF~~Kd_{gln}", lw=lws,alpha = va,subplot=3,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,6]), ylabel = L"ACF~~Y_{lac,glc}", lw=lws,alpha = va,subplot=4,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,7]), ylabel = L"ACF~~Y_{amm,gln}",lw=lws, alpha = va,subplot=5,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,8]), ylabel = L"ACF~~\lambda", lw=lws,alpha = va,xlabel="Lag", subplot=6,label=false,c=corbbf1std104)
#                 plot!(left_margin=4mm,bottom_margin=0mm,  size=(400,800), grid=false)
#                 # annotate!(3., .15,ylabel="ACF", text("BAT-WP using training set A1-Reg1", :left, 10, 8))
#                 # display(plotacf)
#
#             end
#             global casd=casd+1
#             ch = CSV.read(directory*"/"*i, DataFrame)
#             append!(all_chains_bf,[ch])
#         end
#     end
# end
# display(plotacf)
#
#
# # # Classical Bayesian Framework - load all results of exp4 related to RQ4
# directory =  FULL_PATH*"/PUB/empirical_tests/RQ4/exp4"
# all_files = readdir(directory)
# only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
# txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)
# casd=1
#
# all_chains_bf=[]
# all_times=[]
# # plotacf=plot(layout=(6,1),label=false);
# for i in txt_files
#     if occursin("classical_bf_",i )
#         timeRun=parse(Float64, split(i, "_time_")[2][1:9])
#         println(casd ,"| filename: ",i," | time: ",timeRun)
#         if timeRun < 100 #runs above of 100 sec were removed as outliers
#             append!(all_times,timeRun)
#             ch = CSV.read(directory*"/"*i, DataFrame)
#             lws=2.5
#             gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=8);
#             plot!(plotacf,autocor(ch[:,3]), ylabel = L"ACF~~\mu_{max}", lw=lws, alpha = va,subplot=1,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,4]), ylabel = L"ACF~~K_{lysis}", lw=lws,alpha = va,subplot=2,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,5]), ylabel = L"ACF~~Kd_{gln}", lw=lws,alpha = va,subplot=3,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,6]), ylabel = L"ACF~~Y_{lac,glc}", lw=lws,alpha = va,subplot=4,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,7]), ylabel = L"ACF~~Y_{amm,gln}",lw=lws, alpha = va,subplot=5,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,8]), ylabel = L"ACF~~\lambda", lw=lws,alpha = va,xlabel="Lag", subplot=6,label=false,c=corbbf1)
#             plot!(left_margin=4mm,bottom_margin=0mm,  size=(400,800), grid=false)
#             # annotate!(3., .15,ylabel="ACF", text("BAT-WP using training set A1-Reg1", :left, 10, 8))
#             # display(plotacf)
#         end
#         global casd=casd+1
#         ch = CSV.read(directory*"/"*i, DataFrame)
#         append!(all_chains_bf,[ch])
#     end
# end
#
# lws=3
# pltL1 = #vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
# vline(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:left,top_margin=-5mm)
# plot!([NaN], label="CBM", color=c=corbbf1, lw=lws)
# plot!([NaN], label="SBM-BBF1 (NSDS104)", color=c=corbbf1std104, lw=lws)
#
# plot1l1 = plot(plotacf,pltL1,layout=(grid(2,1, heights=[0.92, 0.08])), size=(400, 800))
# display(plot1l1)
#
#
#
#
#
#
#
#
#
#
# corbbf1=:purple2
# corbbf1std104=:royalblue1
#
# # va=0.05
#
#
# # # PUB for BBF1 with NARROW surrogate distribution of exp2
# # load all results of exp2 related to PUB-BBF1
# directory =  FULL_PATH*"/PUB/empirical_tests/RQ2/set100"
# all_files = readdir(directory)
# only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
# txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)
# plotacf=plot(layout=(6,1),label=false);
#
# for  sizedt in [4]
#     all_chains_bf=[]
#     all_times=[]
#     global casd=1
#     for i in txt_files
#         if occursin("chain_bbf1_$(sizedt)_",i )
#             timeRun=parse(Float64, split(i, "_time_")[2][1:9])
#             # println(casd ,"| filename: ",i," | time: ",timeRun)
#             if timeRun < 100 #removed the outliers
#                 append!(all_times,timeRun)
#                 ch = CSV.read(directory*"/"*i, DataFrame)
#                 lws=2.5
#                 gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=8);
#                 plot!(plotacf,autocor(ch[:,3]), ylabel = L"ACF~~\mu_{max}", lw=lws, alpha = va,subplot=1,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,4]), ylabel = L"ACF~~K_{lysis}", lw=lws,alpha = va,subplot=2,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,5]), ylabel = L"ACF~~Kd_{gln}", lw=lws,alpha = va,subplot=3,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,6]), ylabel = L"ACF~~Y_{lac,glc}", lw=lws,alpha = va,subplot=4,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,7]), ylabel = L"ACF~~Y_{amm,gln}",lw=lws, alpha = va,subplot=5,label=false,c=corbbf1std104)
#                 plot!(plotacf,autocor(ch[:,8]), ylabel = L"ACF~~\lambda", lw=lws,alpha = va,xlabel="Lag", subplot=6,label=false,c=corbbf1std104)
#                 plot!(left_margin=4mm,bottom_margin=0mm,  size=(400,800), grid=false)
#                 # annotate!(3., .15,ylabel="ACF", text("BAT-WP using training set A1-Reg1", :left, 10, 8))
#                 # display(plotacf)
#
#             end
#             global casd=casd+1
#             ch = CSV.read(directory*"/"*i, DataFrame)
#             append!(all_chains_bf,[ch])
#         end
#     end
# end
# display(plotacf)
#
#
# # # Classical Bayesian Framework - load all results of exp4 related to RQ4
# directory =  FULL_PATH*"/PUB/empirical_tests/RQ4/exp4"
# all_files = readdir(directory)
# only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
# txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)
# casd=1
#
# all_chains_bf=[]
# all_times=[]
# # plotacf=plot(layout=(6,1),label=false);
# for i in txt_files
#     if occursin("classical_bf_",i )
#         timeRun=parse(Float64, split(i, "_time_")[2][1:9])
#         println(casd ,"| filename: ",i," | time: ",timeRun)
#         if timeRun < 100 #runs above of 100 sec were removed as outliers
#             append!(all_times,timeRun)
#             ch = CSV.read(directory*"/"*i, DataFrame)
#             lws=2.5
#             gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=8);
#             plot!(plotacf,autocor(ch[:,3]), ylabel = "", lw=lws, alpha = va,subplot=1,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,4]), ylabel = "", lw=lws,alpha = va,subplot=2,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,5]), ylabel = "", lw=lws,alpha = va,subplot=3,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,6]), ylabel = "", lw=lws,alpha = va,subplot=4,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,7]), ylabel = "",lw=lws, alpha = va,subplot=5,label=false,c=corbbf1)
#             plot!(plotacf,autocor(ch[:,8]), ylabel = "", lw=lws,alpha = va,xlabel="Lag", subplot=6,label=false,c=corbbf1)
#             plot!(left_margin=4mm,bottom_margin=0mm,  size=(400,800), grid=false)
#             # annotate!(3., .15,ylabel="ACF", text("BAT-WP using training set A1-Reg1", :left, 10, 8))
#             # display(plotacf)
#         end
#         global casd=casd+1
#         ch = CSV.read(directory*"/"*i, DataFrame)
#         append!(all_chains_bf,[ch])
#     end
# end
#
# lws=3
# pltL1 = #vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
# vline(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:left,top_margin=-5mm)
# plot!([NaN], label="CBM", color=c=corbbf1, lw=lws)
# plot!([NaN], label="SBM-BBF1 (NSDS4)", color=c=corbbf1std104, lw=lws)
# plot1l2 = plot(plotacf,pltL1,layout=(grid(2,1, heights=[0.92, 0.08])), size=(400, 800))
# display(plot1l2)
#
#
#
# plotacfall = plot(plot1l1,plot1l2,layout=(grid(1,2, heights=[1, 0])), size=(400, 800))
# # plot!(bottom_margin=-3mm)
#
# display(plotacfall)
# savefig(FULL_PATH*"/PUB/empirical_tests/RQ4/rq4.png")
