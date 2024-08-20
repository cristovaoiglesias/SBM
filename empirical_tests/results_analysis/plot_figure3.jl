using Flux
using Plots
using Distributions
using Turing
using StatsPlots
using Random
using CSV, DataFrames
using Dates
# Tutorial
#  https://dm13450.github.io/2020/12/19/EdwardAndFlux.html







## NN with FLUX


function unpack(nn_params::AbstractVector)
    W1 = reshape(nn_params[1:20], 20, 1); #1x20
    b1 = reshape(nn_params[21:40], 20); #20

    W2 = reshape(nn_params[41:440], 20, 20);# 20x20
    b2 = reshape(nn_params[441:460], 20);#20

    W3 = reshape(nn_params[461:480], 1, 20);#20
    b3 = [nn_params[481]]

    return W1, b1, W2, b2, W3, b3
end

function nn_forward(xs, nn_params::AbstractVector)
    W1, b1, W2, b2, W3, b3 = unpack(nn_params)
    # nn = Chain(Dense(W₁, b₁, tanh), Dense(W₂, b₂))
    nn = Chain(Dense(W1, b1, relu),
                  Dense(W2, b2, relu),
                  Dense(W3, b3))
    return nn(xs)
end

#This Objetive function is the black-box function of this experiment
function BBF(xs,ys,nn_params::AbstractVector)
    W1 = reshape(nn_params[1:20], 20, 1); #1x20
    b1 = reshape(nn_params[21:40], 20); #20

    W2 = reshape(nn_params[41:440], 20, 20);# 20x20
    b2 = reshape(nn_params[441:460], 20);#20

    W3 = reshape(nn_params[461:480], 1, 20);#20
    b3 = [nn_params[481]] #reshape(nn_params[2337:2368], 1);#1

    mlp = Chain(Dense(W1, b1, relu),
                  Dense(W2, b2, relu),
                  Dense(W3, b3))
    x_pred=mlp(xs)
    l=Flux.Losses.mse(x_pred, ys)
    return l
end

# generate the training DT
f(x) = cos(x) + rand(Normal(0, 0.1))

xTrain = collect(-4.75:0.02:1.7)
yTrain = f.(xTrain)
plot(xTrain, yTrain, seriestype=:scatter, label="Train Data")
plot!(xTrain, cos.(xTrain), label="Truth")


# x=xTrain[[Array(1:1:180);Array(150:1:200);Array(280:1:323)]]
# y=yTrain[[Array(1:1:50);Array(150:1:200);Array(280:1:323)]]
x=xTrain[[Array(1:1:180);Array(280:1:323)]]
y=yTrain[[Array(1:1:180);Array(280:1:323)]]
y=[0.09722211392807228, 0.09436825064595704, 0.1564893407890332, 0.002830538986625924, -0.021517349209385216, -0.07002914740947393, -0.24869803382838085, -0.013714626010641509, -0.11867042226165472, -0.1344040064722682, -0.3005850775873382, -0.20233026119097164, -0.1700073065706301, -0.16359296541015508, -0.39493010168048814, -0.42535234925558085, -0.2145716975320166, -0.40074089518296147, -0.3249179125621875, -0.3434451193689504, -0.4210476123139251, -0.2183310437678922, -0.3576765660747292, -0.19401248139473998, -0.3734738423689974, -0.513360797057985, -0.4319489517865463, -0.5768534268245386, -0.5031443374337723, -0.6121063102476376, -0.6512583447936272, -0.6527161299771692, -0.5783575686857442, -0.5958660626917058, -0.670584930055507, -0.7822918904198397, -0.5453413484054292, -0.6421558908021956, -0.6680948867214174, -0.7869010785255073, -0.7202188404215857, -0.9890859377080665, -0.740741766215126, -0.6469781463608849, -0.6511995902254962, -0.8198949385971992, -0.6574882871459912, -0.8725453148500457, -0.7096862832510955, -0.8637311266773278, -0.8795473492696998, -0.8751600303305958, -0.7791980890661846, -0.7424684785407942, -0.7809977391550109, -0.9699138286528701, -0.8696683311772968, -1.0757344538174678, -0.9083450117630174, -1.0418340380584086, -0.8569821894392438, -0.8930894497559589, -0.8448567942113069, -0.9267721884576814, -0.8682798311122498, -0.9080694221888203, -1.0784710375141247, -0.8750055461619332, -1.0268767354024702, -1.0374392732784712, -0.7600662778821382, -0.9472337446140018, -0.8560738166681852, -1.0595974174785312, -1.1917449967161264, -0.8902603662987002, -0.9816734795670936, -0.998521426320988, -1.0605756011718754, -1.0162560183654379, -0.7397813105655939, -1.0631756442532616, -1.1092693289938418, -1.082020337511113, -1.0403731890843728, -0.9781136603352454, -0.9699486329366075, -0.8988255663712261, -0.8747123403571915, -0.9037415002240535, -0.9898239566998631, -0.9815580994175476, -1.0067320626875373, -0.8682635109961128, -0.9733829573929912, -1.104277163610248, -1.0340997365685318, -0.916848963657304, -0.8987223937434501, -1.0350967181731228, -0.7933933930283771, -1.0324593923227017, -1.043456234284624, -0.9080639627983665, -0.7195742665638848, -1.0234174133457565, -0.7925943530150676, -1.176920413891036, -0.9213548221091218, -0.7762938237260427, -0.89949335759924, -0.8859154110891104, -0.764273038103864, -0.6765895819492037, -0.5972163883184931, -0.8696348479659781, -0.8484445070976993, -0.8157506777456894, -0.6524457702259706, -0.738100979164984, -0.6958710936072312, -0.6526644694171282, -0.6033771229201793, -0.5425401421273286, -0.8627108901359564, -0.7060208866794638, -0.4698652654995257, -0.672241121154513, -0.4343814484177865, -0.5735717753171342, -0.5401252793150698, -0.7066155285395836, -0.4206363294404199, -0.4589690857868283, -0.40199676017653674, -0.3727433749907105, -0.5334933764569185, -0.4250419317894505, -0.32965252284093804, -0.34503102811380476, -0.3935539261983544, -0.40220680676460646, -0.1957085214674409, -0.3843604630928206, -0.36507747250176303, -0.1524630268537458, -0.46067816438330195, -0.5618567321288572, -0.15622248391119486, -0.11123407604228, -0.0977701184441494, -0.1672580082411704, -0.19211396775739673, -0.05736440732103794, -0.03943965428145947, -0.1449331439443644, -0.20484344269044444, -0.24794465830433093, -0.11805021203830274, -0.0145471535937882, 0.16305098039375565, 0.0922604041806031, 0.06303118944188878, 0.2225434483498072, 0.15224617393729503, 0.20694232486479847, 0.01698238317533493, 0.12342777323722337, 0.1126611724307766, 0.2261878177430221, 0.34909059550480886, 0.19019147032853823, 0.22469454577869996, 0.124279222073342, 0.3736681821980766, 0.14378571668631934, 0.4309240015346088, 0.23971560952304666, 0.5430743759970067, 0.6010354904998758, 0.5843390742579551, 0.590633599915273, 0.5867642762627886, 0.7582583222465599, 0.593356540095791, 0.42445390799415883, 0.6752860672768497, 0.4797268061563687, 0.6134870165024982, 0.6538769186903366, 0.5716616555074083, 0.5362350001884428, 0.4982574060043867, 0.5211946761179199, 0.5168419204447771, 0.433009236272448, 0.4692617634557157, 0.3967490661224979, 0.37897526710317864, 0.3823453317429101, 0.42586063519508843, 0.1908392294944507, 0.23702457060812707, 0.30538364810995516, 0.31169719597891693, 0.2989417915938425, 0.1077407804533633, 0.19294488858745762, 0.24044497493384476, 0.2755724328965718, 0.12814929537549336, 0.08874817099477915, 0.12129131185304687, 0.06589516387321696, 0.02387359971935208, -0.14450350497995235, 0.07499506178313195, -0.04793878866396995, -0.07064840274676704, -0.07761203068461411, -0.04649858978285776, -0.06512363003656521, -0.17989757685841204, -0.06157870191421286]

estimated_params=[]
all_min_v=[]
print(estimated_params)
ppp=plot(x, y, seriestype=:scatter, label="train Data")





#########
######### Analysing the results
#########

FULL_PATH="" #e.g. "/Users/Thesis" 

ch1 = CSV.read(FULL_PATH*"/SBM/empirical_tests/task2/chain_bnn_umin_29_Jul_024_5:29:47_.csv", DataFrame) #file exp_2.jl

display(ch1)

N = 3000
lp, maxInd = findmax(ch1.lp)
bestParams=Array(ch1[maxInd,3:481+2])
ppp1=plot(x, cos.(x), seriestype=:line, label="True")
plot!(x, Array(nn_forward(hcat(x...), bestParams)'),
      seriestype=:scatter, label="MAP Estimate")
display(ppp1)




xPlot = sort(x)
sp = plot()
paramSample=[]
nsize=500
for i in max(1, (maxInd[1]-nsize)):min(N, (maxInd[1]+nsize))
  # paramSample = map(x-> ch1[x].data[i], params[1:481])
  paramSample=Array(ch1[i,3:481+2])
  plot!(sp, xPlot, Array(nn_forward(hcat(xPlot...), paramSample)'),  alpha = 0.04, label=:none, colour="lightblue")
end
plot!(sp, xPlot, f.(xPlot), seriestype=:scatter, label="Training Data", colour="red")
plot!(xPlot, cos.(xPlot), label="True",w=3, colour="yellow")
plot!(sp, xPlot, Array(nn_forward(hcat(xPlot...), bestParams)'), label="BNN_bestParams", w=3, colour="blue")
display(sp)




xPlot = sort([x;collect(-7:0.1:3.5)])
sp = plot()
nsize=2500
all_prdy=[]
for i in max(1, (maxInd[1]-nsize)):min(N, (maxInd[1]+nsize))
  # paramSample = map(x-> ch1[x].data[i], params[1:481])
  paramSample=Array(ch1[i,3:481+2])
  prdy=Array(nn_forward(hcat(xPlot...), paramSample)')
  append!(all_prdy,[prdy])
  plot!(sp, xPlot,prdy ,  alpha = 0.1,label=:none, colour="lightblue")
end
plot!(sp, sort(x), f.(sort(x)), seriestype=:scatter, label="Training Data", colour="red")
plot!(xPlot, cos.(xPlot),colour="yellow", w=3,label="True")
plot!(sp, xPlot, Array(nn_forward(hcat(xPlot...), bestParams)'), label="BNN_bestParams", w=3, colour="blue")
display(sp)
# savefig(full_path_nm)




using Statistics
using DataFrames


# Function to compute the credible interval (e.g., 95% CI)
function credible_interval(column_data, alpha=0.05)
    sorted_data = sort(column_data)
    lower_idx = Int(ceil(alpha/2 * length(sorted_data)))
    upper_idx = Int(floor((1 - alpha/2) * length(sorted_data)))
    return (sorted_data[lower_idx], sorted_data[upper_idx])
end


function summaryResults(all_prdy)
    # Assuming `all_prdy` is your matrix
    matrix = hcat(all_prdy...)'
    # Convert to DataFrame for easier manipulation (optional)
    df = DataFrame(matrix, :auto)
    # Initialize arrays to store results
    means = []
    std_devs = []
    credible_intervals = []
    # Compute the statistics for each column
    for col in eachcol(df)
        mean_value = mean(col)
        std_dev_value = std(col)
        ci_lower, ci_upper = credible_interval(col)

        push!(means, mean_value)
        push!(std_devs, std_dev_value)
        push!(credible_intervals, (ci_lower, ci_upper))
    end

    # Convert results to DataFrame for better readability
    results = DataFrame(
        mean = means,
        std_dev = std_devs,
        ci_lower = [ci[1] for ci in credible_intervals],
        ci_upper = [ci[2] for ci in credible_intervals]
    )
    # println(results)
    return  results
end


# Function to compute the confidence interval (e.g., 95% CI)
function confidence_interval(column_data, alpha=0.05)
    n = length(column_data)
    mean_value = mean(column_data)
    std_dev_value = std(column_data)
    t_value = quantile(TDist(n-1), 1 - alpha/2)
    margin_of_error = t_value * (std_dev_value / sqrt(n))
    ci_lower = mean_value - margin_of_error
    ci_upper = mean_value + margin_of_error
    return (ci_lower, ci_upper)
end

# Function to summarize results with confidence intervals
function summaryResultsConfI(all_prdy)
    # Assuming `all_prdy` is your matrix
    matrix = hcat(all_prdy...)'
    # Convert to DataFrame for easier manipulation (optional)
    df = DataFrame(matrix, :auto)
    # Initialize arrays to store results
    means = []
    std_devs = []
    confidence_intervals = []
    # Compute the statistics for each column
    for col in eachcol(df)
        mean_value = mean(col)
        std_dev_value = std(col)
        ci_lower, ci_upper = confidence_interval(col)

        push!(means, mean_value)
        push!(std_devs, std_dev_value)
        push!(confidence_intervals, (ci_lower, ci_upper))
    end

    # Convert results to DataFrame for better readability
    results = DataFrame(
        mean = means,
        std_dev = std_devs,
        ci_lower = [ci[1] for ci in confidence_intervals],
        ci_upper = [ci[2] for ci in confidence_intervals]
    )
    # println(results)
    return results
end





using GaussianProcesses

#  Gaussian process
# https://stor-i.github.io/GaussianProcesses.jl/latest/plotting_gps/

xPlot = sort([x;collect(-6:0.1:3.5)])
# Set-up mean and kernel
se = SE(0.185, 0.075)
m = MeanZero()
# Construct and plot GP
gp = GP(x,y,m,se)
plot(gp;  xlabel="gp.x", ylabel="gp.y", title="Gaussian process", legend=false, fmt=:png)
samples = rand(gp, xPlot, 3000)
plot!(xPlot, samples)
# μ, σ² = predict_y(gp,xPlot)

function summaryResultsGP(samples)
    # Assuming `all_prdy` is your matrix
    matrix = samples'
    # Convert to DataFrame for easier manipulation (optional)
    df = DataFrame(matrix, :auto)
    # Initialize arrays to store results
    means = []
    std_devs = []
    credible_intervals = []
    # Compute the statistics for each column
    for col in eachcol(df)
        mean_value = mean(col)
        std_dev_value = std(col)
        ci_lower, ci_upper = credible_interval(col)

        push!(means, mean_value)
        push!(std_devs, std_dev_value)
        push!(credible_intervals, (ci_lower, ci_upper))
    end

    # Convert results to DataFrame for better readability
    results = DataFrame(
        mean = means,
        std_dev = std_devs,
        ci_lower = [ci[1] for ci in credible_intervals],
        ci_upper = [ci[2] for ci in credible_intervals]
    )
    # println(results)
    return  results
end


# Function to summarize results with confidence intervals
function summaryResultsGPConfI(all_prdy)
    # Assuming `all_prdy` is your matrix
    matrix = samples'
    # Convert to DataFrame for easier manipulation (optional)
    df = DataFrame(matrix, :auto)
    # Initialize arrays to store results
    means = []
    std_devs = []
    confidence_intervals = []
    # Compute the statistics for each column
    for col in eachcol(df)
        mean_value = mean(col)
        std_dev_value = std(col)
        ci_lower, ci_upper = confidence_interval(col)

        push!(means, mean_value)
        push!(std_devs, std_dev_value)
        push!(confidence_intervals, (ci_lower, ci_upper))
    end

    # Convert results to DataFrame for better readability
    results = DataFrame(
        mean = means,
        std_dev = std_devs,
        ci_lower = [ci[1] for ci in confidence_intervals],
        ci_upper = [ci[2] for ci in confidence_intervals]
    )
    # println(results)
    return results
end



resultsGP=summaryResultsGP(samples)
# resultsGP=summaryResultsGPConfI(samples)


# xPlot = sort([x;collect(-6:0.1:3.5)])
sp = plot()
nsize=3000
all_prdy=[]
for i in max(1, (maxInd[1]-nsize)):min(N, (maxInd[1]+nsize))
  # paramSample = map(x-> ch1[x].data[i], params[1:481])
  paramSample=Array(ch1[i,3:481+2])
  prdy=Array(nn_forward(hcat(xPlot...), paramSample)')
  append!(all_prdy,[prdy])
  # plot!(sp, xPlot,prdy ,  alpha = 0.1,label=:none, colour="lightblue")
end
results=summaryResults(all_prdy)
# results=summaryResultsConfI(all_prdy)

sp = plot()
# plot!(sp, xPlot, resultsGP[:,3], label="CrIn lower GP", w=3, colour="green")
# plot!(sp, xPlot, resultsGP[:,4], label="CrIn upper GP", w=3, colour="green")
plot!(sp ,xPlot, resultsGP[:,1] , ribbon=(resultsGP[:,1] .- resultsGP[:,3]), fillalpha=0.3, label="95% Credible interval - GP", w=3, colour="green3")
# plot!(sp ,xPlot, resultsGP[:,1] , ribbon=(resultsGP[:,4] .- resultsGP[:,1]), fillalpha=0.3, label="95% Credible interval - GP", w=3, colour="green3")
# plot!(sp, xPlot, results[:,3], label="CrIn lower", w=3, colour="blue")
# plot!(sp, xPlot, results[:,4], label="CrIn upper", w=3, colour="blue")
# plot!(sp ,xPlot, results[:,1] , ribbon=(results[:,1] .- results[:,3]), fillalpha=0.3, label="CrIn lower", w=3, colour="blue")
plot!(sp ,xPlot, results[:,1] , ribbon=(results[:,4] .- results[:,1]), fillalpha=0.3, label="95% Credible interval - BNN from SBM", w=3, colour="blue")
plot!(sp, xPlot, resultsGP[:,1], label="Mean of BNN from SBM", linestyle=:dot, w=3, colour="green")
plot!(sp, sort(x), f.(sort(x)), seriestype=:scatter, label="Noise Data", colour="lightblue")
plot!(xPlot, cos.(xPlot),colour="orange", w=3,label="True y")
plot!(sp, xPlot, results[:,1], label="Mean GP", linestyle=:dot, w=3, colour="blue")
plot!(legend=:bottomleft, size=(600,450),xaxis="x", yaxis="y",grid=false,xtickfontsize=16,ytickfontsize=16,yguidefontsize=18,xguidefontsize=18)
display(sp)
savefig(FULL_PATH*"/SBM/empirical_tests/results_analysis/Figure_3.png")
