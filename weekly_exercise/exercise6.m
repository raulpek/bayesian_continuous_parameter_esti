%% task 1
clc
clearvars
close all
% random
rng(42);
% time grid
N = 200;
t = linspace(0,10,N)';
dt = t(2) - t(1);
M = 8; % num of pairs
t_i = linspace(min(t), max(t), M);
% synthetic data
a0 = -1;
a1 = 0.2;
a2 = -0.015;
p = a0 + a1*t + a2*(t.^2);
theta_true = exp(p); % strictly positive
% K, fixed theta
K_true = build_mat_K(N, theta_true, t, dt);
ell = 0.1; % matern 3/2
sigma_f = 1; % matern 3/2
dist = abs(t - t'); % distance matrix
rr = sqrt(3)*dist/ ell;
C = sigma_f^2*(1 + rr).*exp(-rr);
Lc = chol(C + 1e-8*eye(N),'lower');

sigma_n = 0.03; % noise std
sigma_prior = 1.0;
Sigma = (sigma_n^2)*eye(N);
X_data = Lc*randn(N,M);
Y_raw = K_true*X_data;
Y_data = Y_raw + sigma_n*randn(N,M);

% bayesian neural network
hidden_neurons = 6;
    % prior paths
prior_paths(t, hidden_neurons, sigma_prior, theta_true, N);
inv_sigma_sq = 1 / (sigma_n^2);
log_det_sigma = N*log(sigma_n^2);

p_est = bayes_nn(hidden_neurons, t, X_data, Y_data, N, dt, sigma_prior,...
    inv_sigma_sq, log_det_sigma); % weights+bias
% results
    % t_norm = (t -min(t)) / (max(t) - min(t));
theta_est = psi_nn(p_est, t, hidden_neurons);
K_est = build_mat_K(N, theta_est, t, dt);
Y_est = K_est*X_data;

% error metrics
rmse_theta = sqrt(mean((theta_true - theta_est).^2));
rmse_y = sqrt(mean((Y_raw(:) - Y_est(:)).^2));
% plotting
j = 5;
figure(2);
clf;
subplot(2,2,1);
plot(t, X_data(:,j), "b", "LineWidth", 1.5);
hold on;
plot(t, Y_raw(:,j), "r.", "LineWidth",1.5);
plot(t, Y_data(:,j), "k", "LineWidth",1.5);
xlabel("t");
title("Simulated data (pair 5)");
legend("x_5", " y_5 (noisy)", "Kx_5","Location","best");
grid on;

subplot(2,2,2);
plot(t, theta_true, 'k', "LineWidth",1.5);
hold on;
plot(t, theta_est, 'r--', "LineWidth",1.5);
xlabel("t");
legend("\theta(t) true","ψ^NN(t;w,b)", "Location","best");
title(sprintf("Kernel width, RMSE= %.4f", rmse_theta));

subplot(2,2,3);
plot(t, Y_raw(:,j),"r.");
hold on;
plot(t, Y_data(:,j), "k", "LineWidth",1.5);
plot(t, Y_est(:,j), "b--", "LineWidth",1.5);
xlabel("t");
legend("y_5 (noisy)", "K_true x_5", "K(w,b) x_5", "Location","best  ");
title("Data fit (pair 5)");

subplot(2,2,4);
imagesc(t,t,K_est - K_true);
colorbar;
xlabel("t'");
ylabel("t");
title("K(w,b) - K_{true}");
%% Task 2
clc
clearvars
close all
% random
rng(42);
% time grid
N = 200;
t = linspace(0,10,N)';
dt = t(2) - t(1);
M = 8; % num of pairs
t_i = linspace(min(t), max(t), M);
% synthetic data
a0 = -1;
a1 = 0.2;
a2 = -0.015;
p = a0 + a1*t + a2*(t.^2);
theta_true = exp(p); % strictly positive
% K, fixed theta
K_true = build_mat_K(N, theta_true, t, dt);
ell = 0.1; % matern 3/2
sigma_f = 1; % matern 3/2
dist = abs(t - t'); % distance matrix
rr = sqrt(3)*dist/ ell;
C = sigma_f^2*(1 + rr).*exp(-rr);
Lc = chol(C + 1e-8*eye(N),'lower');

sigma_n = 0.03; % noise std
sigma_prior = 1.0;
Sigma = (sigma_n^2)*eye(N);
X_data = Lc*randn(N,M);
Y_raw = K_true*X_data;
Y_data = Y_raw + sigma_n*randn(N,M);

% bayesian neural network
hidden_neurons = 6;
% prior paths
prior_paths(t, hidden_neurons, sigma_prior, theta_true, N);
inv_sigma_sq = 1 / (sigma_n^2);
log_det_sigma = N*log(sigma_n^2);
p_est = bayes_nn(hidden_neurons, t, X_data, Y_data, N, dt, sigma_prior,...
    inv_sigma_sq, log_det_sigma); % weights+bias
% iterations
n_iter = 20000;
burnIn = 5000;
beta = 0.005;
max_lag = 100;
rng(1);
% results
res = MH(p_est,n_iter, burnIn, beta, t, X_data, Y_data, N,...
    dt, hidden_neurons, sigma_prior,inv_sigma_sq, log_det_sigma); % weights+bias
fprintf("Acceptance rate post-burnIn: %.2f, final stepsize = %.4f\n",...
    res.acc_rate, res.step_size);
chain_full = res.chain_full;
p_samp = res.samp;
n_samp = size(p_samp,2);
% theta(t) per posterior samples
theta_samp = zeros(N, n_samp);
for sm = 1:n_samp
    theta_samp(:, sm) = psi_nn(p_samp(:,sm), t, hidden_neurons);
end
theta_mean = mean(theta_samp,2);
theta_lo = prctile(theta_samp, 2.5, 2);
theta_hi = prctile(theta_samp, 97.5, 2);
theta_map = psi_nn(p_est, t, hidden_neurons); % estimate from prev. task
rmse_theta_mean = sqrt(mean((theta_mean - theta_true).^2));
rmse_theta_map = sqrt(mean((theta_map - theta_true).^2));
cover = mean(theta_true >= theta_lo & theta_true <= theta_hi);

% lets show in table
summary = table(rmse_theta_mean, rmse_theta_map, cover, ...
    'VariableNames', {'rmse_theta_mean','rmse_theta_map','cover'});
disp(summary);

% ACF & ESS
obs_idx = [25, 45, 65, 85, 105, 125];
idx_p = [1, hidden_neurons + 1,...
    2*hidden_neurons+1, 3*hidden_neurons+1];
chains = [p_samp(idx_p,:); theta_samp(obs_idx,:)]';
num_vars = size(chains,2);
acf = zeros(max_lag+1, num_vars);
ess_all = zeros(1, num_vars);
for j = 1:num_vars
    acf(:,j) = simple_acf(chains(:,j), max_lag);
    k_neg = find(acf(2:end, j) < 0, 1);
    if isempty(k_neg)
        k_neg = max_lag + 1;
    end
    ess_all(j) = n_samp / (1+2*sum(acf(2:k_neg,j)));
end
T_ess = table(res.acc_rate, res.step_size, n_samp,...
    min(ess_all), median(ess_all), max(ess_all),...
    min(ess_all)/n_samp,...
    'VariableNames', {'Acc_rate', 'Stepsize',...
        "Nsamples","ESS_min", "ESS_median', "ESS_max",...
        "ESS_min_over_N"});
disp("test ess");
disp(T_ess);
% theta(t) at grid points
tab_theta = table(t(obs_idx), theta_true(obs_idx), ...
    'VariableNames', {'t', 'theta_true'});
disp(tab_theta);
% plotting
figure; % full chains
for j = 1:4
    subplot(4,1,j);
    plot(chain_full(idx_p(j),:));
    xline(burnIn,'r',"LineWidth",2);
    xlabel("iteration");
    ylabel(sprintf("p_{%d}", j));
    if j == 1
        title("MCMC chains");
    end
end

figure; % chains of theta(t)
for j = 1:3
    subplot(3,1,j);
    plot(theta_samp(obs_idx(j),:));
    xline(theta_true(obs_idx(j)),'r',"LineWidth",2);
    xlabel("iteration post-burn-in");
    ylabel(sprintf("theta(t=%.1f)", t(obs_idx(j))));
    if j == 1
        title("Chains of \theta(t) at fixed t");
    end
end

labels = [ ...
    arrayfun(@(i) sprintf("p_{%d}", i), 1:length(idx_p), 'UniformOutput', false), ...
    arrayfun(@(i) sprintf("theta(t=%.1f)", t(obs_idx(i))), 1:length(obs_idx), 'UniformOutput', false) ...
    ];

figure; % ACFs
for j = 1:num_vars
    subplot(num_vars,1,j);
    stem(0:max_lag, acf(:,j), "Marker","none");
    xlabel("lag");
    ylabel(labels{j});
    if j == 1
        title("ACF");
    end
end

X_fill = [t(:);flipud(t(:))];
Y_fill = [theta_lo(:);flipud(theta_hi(:))];
figure; % CI intervals
fill(X_fill,Y_fill,...
    [0.8 0.8 0.8], "EdgeColor","none");
hold on;
plot(t, theta_true, "k", "LineWidth",1.5);
plot(t, theta_mean, "b", "LineWidth",1.5);
plot(t, theta_map, "Color", [1 0.5 0], "LineWidth", 1.5);
xlabel("t");
ylabel("\theta(t)");
legend("95% CI", "true \theta", "post-mean", "MAP (task1)", "Location","southeast");
title("Posterior of \theta(t)")
%% Functions

function K = build_mat_K(N, theta_true, t, dt)
    K = zeros(N,N);
    for i = 1:N
        for j = 1:N
            th = theta_true(j);
            K(i,j) = (1/ (th*sqrt(2*pi))) *...
                exp(-((t(i) - t(j))^2) /...
                (2*th^2))*dt;
        end
    end
end

function prior_paths(t, H, sp, theta_true, n_samples)
    % w,b drawn from gaussian prior
    P = 3*H + 1;
    figure(1);
    clf;
    hold on;
    for k = 1:n_samples
        p_prior = sp*randn(P,1);
        plot(t, psi_nn(p_prior, t, H), 'Color',[0.8 0.8 0.8]);
    end
    plot(t, theta_true, 'r', "LineWidth",1.5);
    xlabel("t");
    ylabel("\theta(t)");
    title("Prior samples of ψ^{NN} & true theta(t)");
end

function res = MH(p0, n_iter, burnIn, beta, t, X_data,...
    Y_data, N, dt, hidden_neurons, sigma_prior, inv_sigma_sq,...
    log_det_sigma)
    P = length(p0);
    chain = zeros(P,n_iter);
    p = p0;
    f = neg_log_post(p,t,X_data, Y_data, N, dt,...
        hidden_neurons, sigma_prior, inv_sigma_sq,...
        log_det_sigma);
    acc_post = 0;
    n_post = n_iter - burnIn;
    acc_win = 0;
    s = 1e-3*sigma_prior;
    for it = 1:n_iter
        % pCn proposal
        z = randn(P,1);
        p_prop = p + s*z;
        f_prop = neg_log_post(p_prop,t,X_data,Y_data,N,dt,...
            hidden_neurons,sigma_prior,inv_sigma_sq,log_det_sigma);
        log_alpha = f - f_prop;
        if isfinite(f_prop) && log(rand) < log_alpha
            p = p_prop;
            f = f_prop;
            acc_win = acc_win + 1;
            if it > burnIn
                % accept post-burnIn
                acc_post = acc_post + 1;
            end
        end
        chain(:,it) = p;

        if it <= burnIn/2 && mod(it,100) == 0
            % adjust step_size
            rate = acc_win / 100;
            acc_win = 0; % reset
            if rate < 0.20
                s = s*0.85;
            elseif rate > 0.45
                s = s*1.15;
            end
        end
    end
    res = struct();
    res.chain_full = chain;
    res.samp = chain(:, burnIn+1:end);
    res.acc_rate = acc_post / n_post;
    res.step_size = s;
end

function wnb = bayes_nn(n_hd_nrs, t, X_data, Y_data, N,dt, sigma_prior,...
    inv_sigma_sq, log_det_sigma)
    P = 3*n_hd_nrs + 1; % num w & b
    opti = optimoptions("fmincon", "Algorithm","interior-point",...
        "HessianApproximation", "lbfgs","MaxIterations",400,...
        "MaxFunctionEvaluations",1e5,"Display","none");
    obj_fn = @(p) neg_log_post(p, t, X_data, Y_data,...
        N, dt, n_hd_nrs,sigma_prior, inv_sigma_sq, log_det_sigma);
    best_f = inf;
    for k = 1:5
        p0 = 0.5*randn(P,1);
        [p_k, f_k] = fmincon(obj_fn,p0, [],[],[], [],...
            [],[],[], opti);
        if f_k < best_f
            best_f = f_k;
            wnb = p_k;
        end
    end
end

function theta = psi_nn(p,t,H)
    idx = 1;
    W1 = reshape(p(idx:idx + H - 1), [H, 1]);
    idx = idx + H;
    b1 = reshape(p(idx:idx + H - 1), [H, 1]);
    idx = idx + H;
    W2 = reshape(p(idx:idx + H - 1), [1, H]);
    b2 = p(idx + H);
    tNorm = (t - min(t)) / (max(t) - min(t));
    hiddenOutput = tanh(W1 * tNorm' + b1);
    rawTheta = W2 * hiddenOutput + b2;
    theta = 0.1 + log1p(exp(rawTheta(:)));
end

function ng_lg_post = neg_log_post(params, t, X_data,Y_data, N, dt,...
    hidden_neurons, sigma_prior, inv_sigma_sq, log_det_sigma)
    % unpack params
    idx = 1;
    W1 = reshape(params(idx:idx + hidden_neurons-1), [hidden_neurons,1]);
    idx = idx + hidden_neurons;
    b1 = reshape(params(idx:idx + hidden_neurons-1), [hidden_neurons,1]);
    idx = idx + hidden_neurons;
    W2 = reshape(params(idx:idx + hidden_neurons-1), [1, hidden_neurons]);
    idx = idx + hidden_neurons;
    b2 = params(idx);
    % normalize t
    t_norm = (t - min(t)) / (max(t) - min(t));
    % Psi_NN(t;w,b) evaluation
    hidden_out = tanh(W1 * t_norm' + b1);
    raw_theta = W2 *hidden_out + b2;
    theta_nn = 0.1 + log(1 + exp(raw_theta(:)));
    % theta_nn = max(0.1,softval);
    theta_nn = theta_nn(:);
    % K_(w,b)
    K_wb = build_mat_K(N, theta_nn, t, dt);
    % log likelihood sum
    log_like = 0;
    M = size(X_data,2);
    X_data = num2cell(X_data,1);
    Y_data = num2cell(Y_data,1);
    for i = 1:M
        xi = X_data{i};
        yi = Y_data{i};
        resi = yi - K_wb *xi;
        quad_form = sum(resi.^2) * inv_sigma_sq;
        log_like = log_like - 0.5*(N*log(2*pi) +...
            log_det_sigma + quad_form);
    end
    % log prior from gaussian prior on w
    n_params = length(params);
    log_prior = -0.5*sum((params / sigma_prior).^2) -...
        n_params*log(sigma_prior*sqrt(2*pi));
    % negative log posterior
    ng_lg_post = -1*(log_like + log_prior);
end

function a = simple_acf(x, maxlag)
    x = x(:) - mean(x);
    c = zeros(maxlag+1,1);
    for k = 0:maxlag
        c(k+1) = sum(x(1:end-k) .* x(1+k:end));
    end
    a = c / c(1);
end