using GaussianMarkovRandomFields
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using FerriteViz
using Distributions
using Random, Statistics
using LinearAlgebra
using Plots
using XLSX
using Proj
using PythonCall
using CairoMakie
using Distances
using SpecialFunctions
using BenchmarkTools
# parameter struct
struct param_tbl
    range::Float64
    tau::Float64
    sigma_f::Float64
    nu::Int64
    sigma_n::Float64
end

# 2) Real-live 2D dataset
function load_data()
    df2 = XLSX.openxlsx("air_pollution.xlsx")
    sheet = df2["air_pollution"]
    raw_data = sheet[2:145, 15:17] # ignore the header
    return raw_data
end

function trans_data(raw_data)
    lati = [parse(Float64, replace(string(val), ',' => '.')) for val in raw_data[:,1]]
    longi = [parse(Float64, replace(string(val), ',' => '.')) for val in raw_data[:,2]]
    pm25 = [parse(Float64, replace(string(val), ',' => '.')) for val in raw_data[:,3]]
    # transform data to x,y coordinates
    trans = Proj.Transformation("EPSG:4326", "EPSG:32634", always_xy=true)
    xy_coord = [trans((lo,la)) for (lo, la) in zip(longi, lati)]
    x_m = first.(xy_coord)
    y_m = last.(xy_coord)
    X = hcat(x_m, y_m)
    return X, pm25
end

function tr_ts_split(X, pm25)
    # train and test indices
    train_idx = randsubseq(1:size(X,1), 0.8)
    test_idx = [i for i in 1:size(X,1) if isempty(searchsorted(train_idx, i))]
    X_train = X[train_idx, :]
    X_test = X[test_idx,:]
    z_tr_mat = pm25[train_idx,:]
    z_ts_mat = pm25[test_idx,:]
    z_train = vec(z_tr_mat)
    z_test = vec(z_ts_mat)
    return X_train, X_test, z_train, z_test
end

function data_gen(N::Int64, x_span::Float64, y_span::Float64)
    X_raw = rand(2,N) .* [x_span; y_span]
    X = X_raw'
    z = randn(N)*2.0 .+ 50.0
    return X,z

end

function GMRF_func(params, X, X_train, X_test, z_train, z_test)
    latent_model = MaternModel(X; smoothness=params.nu)
    u_matern = latent_model(τ=params.tau, range=params.range)
    # observation model and posterior
    obs_model = PointEvaluationObsModel(latent_model.discretization, X_train, Distributions.Normal)
    obs_likelihood = obs_model(z_train, σ = params.sigma_n)
    t = @belapsed begin
        gaussian_approximation($u_matern, $obs_likelihood)
    end
    # evaluation
    println("TIME GMRF: $(round(t, digits=12))")
    u_condi = gaussian_approximation(u_matern, obs_likelihood)
    mu_u_condi = mean(u_condi)
    std_u_condi = std(u_condi)
    obs_model_test = PointEvaluationObsModel(latent_model.discretization, X_test, Distributions.Normal)
    pred_dist_train = conditional_distribution(obs_model, mu_u_condi; σ = params.sigma_n)
    pred_dist_test = conditional_distribution(obs_model_test, mu_u_condi; σ = params.sigma_n)
    rmse = (a,b) -> sqrt(mean(a .-b) .^2)
    tr_rmse = rmse(mean(pred_dist_train), z_train)
    ts_rmse = rmse(mean(pred_dist_test), z_test)
    dh = latent_model.discretization.dof_handler
    return tr_rmse, ts_rmse, dh, mu_u_condi, std_u_condi
end

function save_plots(X_test, dh, mu_u_condi, mu_u_std, post_mean_GP, post_std_GP,  test_size, gp_times, gmrf_times)
    # define color ranges
    vmin = min(maximum(post_mean_GP), minimum(mu_u_condi))
    vmax = max(maximum(post_mean_GP), maximum(mu_u_condi))
    vmin_std = min(maximum(post_std_GP), minimum(mu_u_std))
    vmax_std = max(maximum(post_std_GP), maximum(mu_u_std))
    #create png
    CairoMakie.activate!()
    ds_mean = FEData(dh, mu_u_condi)
    # posterior mean GMRF
    println("posterior mean plot gmrf")
    fig1 = FerriteViz.solutionplot(ds_mean; colormap =:coolwarm)
    ax_fig1 = fig1.axis
    ax_fig1.xlabel = "x"
    ax_fig1.ylabel = "y" 
    CairoMakie.save("plots/air_pollu_mean.png", fig1)
    # posterior std GMRF
    println("posterior std plot gmrf")
    ds_std = FEData(dh, mu_u_std)
    fig2 = FerriteViz.solutionplot(ds_std; colormap =:coolwarm)
    ax_fig2 = fig2.axis
    ax_fig2.xlabel = "x"
    ax_fig2.ylabel = "y" 
    CairoMakie.save("plots/air_pollu_std.png", fig2)
    # posterior mean GP
    println("posterior mean plot gp")
    fig3 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig3[1,1], aspect= CairoMakie.DataAspect(), title="GP posterior mean", xlabel="x", ylabel="y")
    plt1 = CairoMakie.tricontourf!(ax1, X_test[:,1], X_test[:,2], post_mean_GP; levels=30, colormap =:coolwarm, colorrange=(vmin,vmax))
    CairoMakie.Colorbar(fig3[1,2], plt1,; label="Posterior mean")
    CairoMakie.save("plots/gp_posterior_mean.png", fig3)
    # posterior std GP
    println("posterior std plot gp")
    fig4 = CairoMakie.Figure()
    ax2 = CairoMakie.Axis(fig4[1,1], aspect= CairoMakie.DataAspect(), title="GP posterior std", xlabel="x", ylabel="y")
    plt2 = CairoMakie.tricontourf!(ax2, X_test[:,1], X_test[:,2], vec(post_std_GP); levels=30,  colormap =:coolwarm, colorrange=(vmin_std,vmax_std))
    CairoMakie.Colorbar(fig4[1,2], plt2,; label="Posterior std")
    CairoMakie.save("plots/gp_posterior_std.png", fig4)
    # time comparison
    println("time comparison plot")
    gp_times_ms = gp_times * 1000
    gmrf_times_ms = gmrf_times * 1000
    fig5 = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig5[1,1], xlabel = "N", ylabel = "ms", title = "GP vs GMRF",yscale=log10)
    CairoMakie.lines!(ax, test_size, gp_times_ms, label="GP", color= :red, linewidth=2)
    CairoMakie.lines!(ax, test_size, gmrf_times_ms, label="GMRF", color = :green, linewidth=2)
    CairoMakie.axislegend(ax, position =:rb)
    CairoMakie.save("plots/time_comparison.png", fig5)
end

# Create a GP model with the same MaternModel
function matern_mat(R, nu, sigma_f, range)
    sigma_l = range / 2
    arg = sqrt(2*nu) .* R ./ sigma_l
    # general matern formula
    K = zeros(size(R))
    i = arg .> 0
    K[i] .= (sigma_f^2) * (2^(1-nu) / gamma(nu)) .* (arg[i] .^nu) .* besselk.(nu, arg[i])
    K[.!i] .= sigma_f^2
    return K
end
function GP_func(params, X_train, X_test, z_train, z_test)
    # obs-to-obs covariance
    dist_trtr = pairwise(Euclidean(), X_train, X_train; dims=1)
    dist_no_zeros_trtr = max.(dist_trtr, eps(eltype(dist_trtr)))
    K_trtr = matern_mat(dist_no_zeros_trtr, params.nu, params.sigma_f, params.range)
    # test-to-obs cross-covariance
    dist_trts = pairwise(Euclidean(), X_test, X_train; dims=1)
    dist_no_zeros_trts = max.(dist_trts, eps(eltype(dist_trts)))
    K_trts =  matern_mat(dist_no_zeros_trts, params.nu, params.sigma_f, params.range)
    # test-to-test
    dist_tsts = pairwise(Euclidean(), X_test, X_test; dims=1)
    dist_no_zeros_tsts = max.(dist_tsts, eps(eltype(dist_tsts)))
    K_tsts =  matern_mat(dist_no_zeros_tsts, params.nu, params.sigma_f, params.range)
    # prediction
    println("GP:")
    L = cholesky(K_trtr + params.sigma_n^2 * I)
    alfa = L \ z_train
    mean_test_GP = K_trts * alfa # posterior mean
    # tmiming test
    t = @belapsed begin
        K = matern_mat($dist_no_zeros_trtr,$params.nu, $params.sigma_f, $params.range)
        L = cholesky(K + $params.sigma_n^2 *I)
        alfa = L \ $z_train
    end
    println("TIME GP: $(round(t, digits=12))")
    # posterior marginal variance
    v = L \ K_trts' 
    var_test = diag(K_tsts) - sum(K_trts .* v', dims=2)
    gp_std = sqrt.(max.(var_test, 0.0))
    mean_train_GP = K_trtr *(L \ z_train)
    rmse = (a,b) -> sqrt(mean(a .-b) .^2)
    rmse_train_GP = rmse(mean_train_GP, z_train)
    rmse_test_GP = rmse(mean_test_GP, z_test)
    return  rmse_train_GP, rmse_test_GP, mean_test_GP, gp_std
end

function run_tests(arg, plot_on)
    params = param_tbl(100_000.0, 1.0, 1.0, 1,  1.0)
    # load the data
    raw_data = load_data()
    # transform data
    X, pm25 = trans_data(raw_data)
    # train_test data
    X_train, X_test, z_train, z_test = tr_ts_split(X, pm25)
    # GMRF
    tr_rmse_GMRF, ts_rmse_GMRF, dh, mu_u_condi, mu_u_std = GMRF_func(params, X, X_train, X_test, z_train, z_test)
    # GP
    tr_rmse_GP, ts_rmse_GP, post_mean_GP, post_std_GP = GP_func(params, X_train, X_test, z_train, z_test)
    print("GMRF train RMSE:")
    println(tr_rmse_GMRF)
    print("GMRF test RMSE:")
    println(ts_rmse_GMRF)
    print("GP train RMSE:")
    println(tr_rmse_GP)
    print("GP test RMSE:")
    println(ts_rmse_GP)
    # let's try bigger datasets
    if arg == true
        println("Testing different datasizes")
        test_sizes = [500, 1500, 3000, 4500]
        gmrf_times = Float64[]
        gp_times = Float64[]
        # spatial bounds
        x_span = 600_000.0
        z_span = 500_000.0
        for test_sz in test_sizes
            println("===== Testing N: $test_sz ===")
            # synthetic data generation
            X_synth, z_synth = data_gen(test_sz, x_span, z_span)
            # standard gp
            D = pairwise(Euclidean(), X_synth, X_synth; dims=1)
            K = matern_mat(D, 1.0, 1.0, 150_000.0)
            time_gp = @belapsed begin
                L = cholesky($K + (0.15^2 * I))
                L \ $z_synth
            end
            push!(gp_times, time_gp)
            println("GP TIME : $(round(time_gp, digits=12)) seconds")
            # GMRF time_gp
            model = MaternModel(X_synth; smoothness = 1)
            u_matern = model(τ=1.0, range=150_000.0)
            obs_model = PointEvaluationObsModel(model.discretization, X_synth, Distributions.Normal)
            obs_like = obs_model(z_synth;σ=0.15)
            # factorization
            time_gmrf = @belapsed begin
                gaussian_approximation($u_matern, $obs_like)
            end
            push!(gmrf_times, time_gmrf)
            println("GMRF TIME : $(round(time_gmrf, digits=12)) seconds")
        end
        # create plots
        if plot_on
            println("¤¤¤¤ Creating and saving plots ¤¤¤¤")
            save_plots(X_test, dh, mu_u_condi, mu_u_std, post_mean_GP, post_std_GP, test_sizes, gp_times, gmrf_times)
        end

    end
end

# run the experiments
run_tests(true, false)



