clc
clearvars
close all
% space
h = 0.01;
i = linspace(0,100, 101);
t1 = (i*h)';
N = length(t1);
% Matern covariance 1D
sigma_f = 1.0; % signal variance
sigma_l = 0.45; % length-scale, l true
sigma_n = 0.1; % noise
nu_true = 1.75; % ν = 1.75.
C_true = matern_own(t1,[sigma_l,nu_true],sigma_f);
% Cholesky factorization
[U_up, ~] = chol(C_true + 1e-6*eye(size(C_true)));
U1 = U_up'; % lower triangular
% Ground Truth
rng(42);
W_omega = randn(N,1);
X_omega = U1*W_omega;
y_obs = X_omega + sigma_n *randn(N,1);
obs_idx = (1:N)';
% Estimation phase
% priors and number of chains
n_mcmc = 30000;

% X_curr = zeros(N,1);
ini_sigma_n = 0.3;
samples_sigma_n = zeros(1,n_mcmc);
ini_nu = 0.5;

u_pcn = zeros(N, n_mcmc); % proposals
u_pcn(:,1) = zeros(N,1);
samples_X = zeros(N,n_mcmc); % accepted states

C2 = matern_own(t1,[sigma_l, nu_true], sigma_f);
[U_fixed, ~] = chol(C2 + 1e-8*eye(N),'lower');

% try both priors
priors = ["uni", "log_n"];
K = [];
res = struct();
for prior = priors
    % 1) estimate jointly noise var
    % and all the states
    res.(prior).t1_1 = pcn_algo_joint1(u_pcn,U_fixed,...
        y_obs, N,...
        obs_idx, ini_sigma_n, "task1", K, prior);
    % 2) estimate jointly noise var,nu
    res.(prior).t1_2 = pcn_algo_joint2(u_pcn,C2,...
        y_obs, N,...
        obs_idx, sigma_n, [ini_nu, sigma_f, sigma_l],...
        t1,"task1", K, prior);
end

% plotting first both uniform
results1 = res.uni.t1_1;
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_sigma"));
sig_n_full = results1("full_sigma");
u_full = results1("full_u");
X_full = U_fixed*u_full;
sigma_samp = results1("sigma_samp");
X_samp = results1("results");
x_mean = results1("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 1.1: MCMC chain for sigma_n");
grid on;

subplot(2,1,1); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 1.1: MCMC chain for 1st state");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 1.1: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_omega, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_omega).^2,2));
rmse_trace = sqrt(mean((X_samp-X_omega).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("MCMC");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;

clear sig_n_full sigma_samp...
    X_samp x_mean x_low x_up...
    results1 u_full X_full rmse_t...
    rmse_trace
% 1.1 both log-normal
results1 = res.log_n.t1_1;
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_sigma"));
sig_n_full = results1("full_sigma");
u_full = results1("full_u");
X_full = U_fixed*u_full;
sigma_samp = results1("sigma_samp");
X_samp = results1("results");
x_mean = results1("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 1.1: MCMC chain for sigma_n");
grid on;

subplot(2,1,1); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 1.1: MCMC chain for 1st state");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 1.1: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_omega, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_omega).^2,2));
rmse_trace = sqrt(mean((X_samp-X_omega).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("MCMC");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;


% 2) estimate jointly noise var,nu
clear sig_n_full sigma_samp...
    X_samp x_mean x_low x_up...
    results1 u_full X_full rmse_t...
    rmse_trace


% both unifrom
results2 = res.uni.t1_2;
fprintf('Acceptance Rate: %.2f %%\n', results2("acc_rate_sigma"));
fprintf('Acceptance Rate: %.2f %%\n', results2("acc_rate_nu"));
sig_n_full = results2("full_sigma");
u_full = results2("full_u");
X_full = U_fixed*u_full;
sigma_samp = results2("sigma_samp");
nu_samp = results2("nu_samp");
X_samp = results2("results");
x_mean = results2("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 1.2: MCMC chain for sigma_n");
grid on;

subplot(2,1,2); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 1.2: MCMC chain for 1st state");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 1.2: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_omega, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("Task 1.2: 95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_omega).^2,2));
rmse_trace = sqrt(mean((X_samp-X_omega).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("MCMC");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;

clear sig_n_full sigma_samp...
    X_samp x_mean x_low x_up...
    results1 u_full X_full rmse_t...
    rmse_trace

results2 = res.log_n.t1_2;
fprintf('Acceptance Rate: %.2f %%\n', results2("acc_rate_sigma"));
fprintf('Acceptance Rate: %.2f %%\n', results2("acc_rate_nu"));
sig_n_full = results2("full_sigma");
u_full = results2("full_u");
X_full = U_fixed*u_full;
sigma_samp = results2("sigma_samp");
nu_samp = results2("nu_samp");
X_samp = results2("results");
x_mean = results2("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 1.2: MCMC chain for sigma_n");
grid on;

subplot(2,1,2); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 1.2: MCMC chain for 1st state");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 1.2: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_omega, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("Task 1.2: 95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_omega).^2,2));
rmse_trace = sqrt(mean((X_samp-X_omega).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("MCMC");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;

%% Task 2
clc
clearvars
close all

% space
h = 0.01;
i = linspace(0,100, 101);
t1 = (i*h)';
N = length(t1);
% Truth
X_true = 0.8*sin(2*pi*t1) + 0.4*cos(3*pi*t1)+...
    0.5*exp(-50*(t1-0.65).^2);

% now let's apply the forward operator
lim1 = -0.2;
lim2 = 0.2;
K = forward_mat(t1, lim1,lim2);

rng(42); % random seed
y_raw = K*X_true;
noise = 0.1*randn(size(X_true)); % true sig_n = 0.1
y_noisy = y_raw + noise;
% Chain setup
n_mcmc = 30000;
burnin = 5000;
thin = 10;

% initial guesses
sigma_n_curr = 0.01;
nu_true = 1.75;
sigma_l = 0.45; % fixed
sigma_f = 1.0;

u_pcn = zeros(N,n_mcmc);

% fixed nu, estimate states and sigma_n

C_init = matern_own(t1, [sigma_l, nu_true], sigma_f);
[U_up, ~] = chol(C_init + 1e-8*eye(size(C_init)));
U1 = U_up'; % lower triangular

% try both priors
priors = ["uni", "log_n"];
res = struct();
ini_sigma_n = 0.3;
ini_nu = 0.5;
for prior = priors
    % 1) estimate jointly noise var
    % and all the states
    res.(prior).t2_1 = pcn_algo_joint1(u_pcn,U1,...
        y_noisy, N, [],...
        ini_sigma_n, "task2.1", K, prior);
    % 2) estimate jointly noise var,nu
    res.(prior).t2_2 = pcn_algo_joint2(u_pcn, C_init,...
        y_noisy,N, ...
        [], ini_sigma_n,[ini_nu,sigma_l,sigma_f],...
        t1, "task2.2",K, prior);
end

% uniform plots
results1 =  res.uni.t2_1;
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_sigma"));
sig_n_full = results1("full_sigma");
u_full = results1("full_u");
% X_full = U1*u_full;
sigma_samp = results1("sigma_samp");
X_samp = results1("results");
x_mean = results1("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 2.1: MCMC chain for sigma_n");
grid on;

subplot(2,1,1); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 2.1: MCMC chain for 1st state");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 2.1: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_true, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("Task 2.1: 95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_true).^2,2));
rmse_trace = sqrt(mean((X_samp-X_true).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;

% lognormal plots
clear results1 sig_n_full sigma_samp...
    X_samp x_mean x_low x_up...
    results u_full X_full rmse

results1 =  res.log_n.t2_1;
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_sigma"));
sig_n_full = results1("full_sigma");
u_full = results1("full_u");
X_full = U1*u_full;
sigma_samp = results1("sigma_samp");
X_samp = results1("results");
x_mean = results1("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 2.1: MCMC chain for sigma_n");
grid on;

subplot(2,1,2); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 2.1: MCMC chain for 1st state");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 2.1: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_true, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("Task 2.1: 95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_true).^2,2));
rmse_trace = sqrt(mean((X_samp-X_true).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;


% estimate states, sigma_n, nu
clear results1 sig_n_full sigma_samp...
    X_samp x_mean x_low x_up...
    results u_full X_full rmse

results1 = res.uni.t2_2;
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_sigma"));
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_nu"));
sig_n_full = results1("full_sigma");
nu_full = results1("full_nu");
X_full = results1("results");
sigma_samp = results1("sigma_samp");
nu_samp = results1("nu_samp");
X_samp = results1("results");
x_mean = results1("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(3,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 2.2: MCMC chain for sigma_n");
grid on;

figure; % MCMC chain for nu
subplot(3,1,1);
plot(nu_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("nu");
title("Task 2.2: MCMC chain for nu");
grid on;

subplot(3,1,2); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 2.2: MCMC chain for 1st state");
grid on;

figure; % MCMC chain for sigma_n
subplot(3,1,3);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 2.2: MCMC chain for sigma_n");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 2.2: Posterior marginal of sigma_n");
grid on;

figure; % posterior marginal of nu
histogram(nu_samp, 50, 'Normalization','pdf');
xlabel("\nu");
ylabel("Density");
title("Task 2.2: Posterior marginal of \nu");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_true, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("Task 2.2: 95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_true).^2,2));
rmse_trace = sqrt(mean((X_samp-X_true).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;

% log-normal
clear results1 sig_n_full sigma_samp...
    X_samp x_mean x_low x_up...
    results u_full X_full rmse

results1 = res.log_n.t2_2;
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_sigma"));
fprintf('Acceptance Rate: %.2f %%\n', results1("acc_rate_nu"));
sig_n_full = results1("full_sigma");
nu_full = results1("full_nu");
X_full = results1("results");
sigma_samp = results1("sigma_samp");
nu_samp = results1("nu_samp");
X_samp = results1("results");
x_mean = results1("mean"); % posterior mean
x_low = prctile(X_samp, 2.5, 2);
x_up = prctile(X_samp, 97.5, 2);

figure; % MCMC chain for sigma_n
subplot(3,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 2.2: MCMC chain for sigma_n");
grid on;

figure; % MCMC chain for nu
subplot(3,1,2);
plot(nu_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("nu");
title("Task 2.2: MCMC chain for nu");
grid on;

subplot(3,1,3); % MCMC for 1st state
plot(X_samp(1,:), "LineWidth",1);
xlabel("Post-burn-in iteration");
ylabel("x_1");
title("Task 2.2: MCMC chain for 1st state");
grid on;

figure; % MCMC chain for sigma_n
subplot(2,1,1);
plot(sig_n_full,"LineWidth",1);
xline(5000,"--", "Burn-in","LineWidth",1.2);
xlabel("MCMC iteration");
ylabel("sigma_n");
title("Task 2.2: MCMC chain for sigma_n");
grid on;

figure; % posterior marginal of sigma_n
histogram(sigma_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 2.2: Posterior marginal of sigma_n");
grid on;

figure; % posterior marginal of nu
histogram(nu_samp, 50, 'Normalization','pdf');
xlabel("sigma_n");
ylabel("Density");
title("Task 2.2: Posterior marginal of sigma_n");
grid on;

figure; % p95 state
fill([t1;flipud(t1)],...
    [x_low; flipud(x_up)],...
    [0.85 0.85 0.85],...
    'EdgeColor','none');
hold on;
plot(t1, X_true, "k--", "LineWidth",1.5);
plot(t1, x_mean,'b', "LineWidth",1.5);
xlabel("t");
ylabel("X(t)");
legend("95% posterior interval",...
    "True state", "Posterior mean");
title("Task 2.2: 95% posterior interval");
grid on;
hold off;

rmse_t = sqrt(mean((X_samp-X_true).^2,2));
rmse_trace = sqrt(mean((X_samp-X_true).^2));

figure; % pointwise RMSE plot for sigma_n
subplot(1,2,1);
plot(t1, rmse_t, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over t");
grid on;

subplot(1,2,2);
plot(rmse_trace, "LineWidth",1.5);
xlabel("t");
ylabel("RMSE");
title("Task 2.2: RMSE over MCMC");
grid on;
%% Functions

% Matern Covariance
function K = matern_own(X, hyper_p, sigma_f)
    sigma_l = hyper_p(1);
    nu = hyper_p(2);
    dist = pdist2(X,X);
    % zero-distance handling
    dist(dist==0) = eps;
    % scaling factor
    arg = sqrt(2*nu)*dist / sigma_l;
    % general matern
    K = (sigma_f^2)*...
        (2^(1-nu) / gamma(nu)) *...
        (arg.^nu) .* besselk(nu,arg);
    % variance on the diagonal
    K(logical(eye(size(K)))) = sigma_f^2;
end


% PCN1
function res = pcn_algo_joint1(u_pcn,L_curr,...
    y_obs,N, obs_idx, sigma_n, command,...
    K, prior)
    nums = size(u_pcn,2); % n_mcmc
    n_obs = length(y_obs);
    accepted = 0;
    acc_sigma_n = 0;
    beta = 0.15;
    burnIn = 5000;
    samples_sigma_n = zeros(1, nums+1);
    tuning_sig = 0.05; % sigma_tuning
    for i = 1:nums-1
        % proposal of latent WN u
        xi = randn(N,1);
        u_prop = sqrt(1-beta^2)*u_pcn(:,i) + beta*xi;
        % acceptance rate to u
        lg_a_u = -phi_calc(u_prop,y_obs,obs_idx,sigma_n, L_curr, K)...
            + phi_calc(u_pcn(:,i),y_obs,obs_idx,sigma_n, L_curr, K);
        if log(rand()) < lg_a_u
            u_pcn(:,i+1) = u_prop;
            accepted = accepted + 1;
        else
            % reject
            u_pcn(:,i+1) = u_pcn(:,i);
        end
        % update sigma_n
        lg_sigma_n_prop = log(sigma_n) + tuning_sig *randn();
        sigma_n_prop = exp(lg_sigma_n_prop);
        lg_sigma_n = log(sigma_n);
        X_curr = L_curr*u_pcn(:,i+1);
        if command == "task2.1"
            y_curr = K*X_curr;
            resi = y_obs - y_curr;
            nll_curr = (0.5*sum(resi.^2) /...
                (sigma_n^2)) + ((n_obs/2)*log(sigma_n^2));
            nll_prop = (0.5*sum(resi.^2) /...
                (sigma_n_prop^2)) + ((n_obs/2)*log(sigma_n_prop^2));
        else
            nll_curr = (0.5*sum((y_obs - X_curr(obs_idx)).^2) /...
                (sigma_n^2)) + ((n_obs/2)*log(sigma_n^2));
            nll_prop = (0.5*sum((y_obs - X_curr(obs_idx)).^2) /...
                (sigma_n_prop^2)) + ((n_obs/2)*log(sigma_n_prop^2));
        end
        % prior sigma
        lg_prior_curr = log_prior(sigma_n, "sigma", prior);
        lg_prior_prop = log_prior(sigma_n_prop,"sigma", prior);
        % evaluate
        lp_a_sig_n = (-nll_prop) - (-nll_curr) +...
            (lg_prior_prop - lg_prior_curr)...
            + (lg_sigma_n_prop - lg_sigma_n);
        if log(rand()) < lp_a_sig_n
            sigma_n = sigma_n_prop;
            acc_sigma_n = acc_sigma_n + 1;
        end
        samples_sigma_n(i+1) = sigma_n;
    end
    % sample processing
    u_pcn_samp = u_pcn(:, (burnIn+1):end);
    sigma_n_samp = samples_sigma_n(:, (burnIn+1):end);
    
    X_pcn_samp = L_curr * u_pcn_samp;
    mu_pcn = mean(X_pcn_samp,2);
    mu_sigma_n = mean(sigma_n_samp,2);

    std_pcn = std(X_pcn_samp,0,2);
    std_sigma_n = std(sigma_n_samp,0,2);

    acc_rate = (accepted / (nums-1))*100;
    acc_rate_sigma_n = (acc_sigma_n /(nums-1))*100;
    % create output
    keys  = ["results", "mean", "std",...
        "acceptance rate","sigma_samp",...
        "sigma_mean", "sigma_std",...
        "full_u", "full_sigma", "acc_rate_sigma"];
    values = {X_pcn_samp, mu_pcn,...
        std_pcn, acc_rate,...
        sigma_n_samp, mu_sigma_n,...
        std_sigma_n, u_pcn,...
        samples_sigma_n, acc_rate_sigma_n};
    res = containers.Map(keys,values);
end

% PCN2
function res = pcn_algo_joint2(u_pcn,C_curr,...
    y_obs,N, obs_idx, sigma_n, params, t1,...
    command, K, prior)
    nums = size(u_pcn,2); % n_mcmc
    nu_curr = params(1);
    sigma_l = params(2); % fixed sigma_l
    sigma_f = params(3); % fixed sigma_f
    uni_lim = [0.1,5];
    n_obs = length(y_obs);
    accepted = 0;
    accepted_nu = 0;
    acc_sigma_n = 0;
    beta = 0.15;
    burnIn = 5000;
    samples_sigma_n = zeros(1, nums+1);
    samples_sigma_n(1) = sigma_n;
    samples_nu = zeros(1, nums+1);
    samples_nu(1) = nu_curr;
    X_all = zeros(N, nums);
    tuning_sig = 0.05; % sigma_tuning
    delta_nu = 0.1;
    L_curr = chol(C_curr + 1e-6*eye(N),'lower');
    for i = 1:nums-1
        % proposal nu
        nu_prop = nu_curr + (rand() - 0.5)*2 *delta_nu;
        lp_nu_prop = log_prior(nu_prop,"nu",prior);
        lp_nu_curr = log_prior(nu_curr, "nu", prior);
        lg_a_nu = -Inf;
        if lp_nu_prop > -Inf
            C_prop = matern_own(t1, [sigma_l, nu_prop], sigma_f);
            [L_prop, p] = chol(C_prop + 1e-6*eye(N),'lower');
            if p == 0
                lg_a_nu = -phi_calc(u_pcn(:,i),y_obs,obs_idx,sigma_n,L_prop,K)...
                    + phi_calc(u_pcn(:,i),y_obs,obs_idx,sigma_n,L_curr,K)...
                    + lp_nu_prop - lp_nu_curr;
            end
        end
        if log(rand())< lg_a_nu
            nu_curr = nu_prop;
            C_curr = C_prop;
            L_curr = L_prop;
            accepted_nu = accepted_nu + 1;
        end
        samples_nu(i+1) = nu_curr;
        % proposal of latent WN u
        xi = randn(N,1);
        u_prop = sqrt(1-beta^2)*u_pcn(:,i) + beta*xi;
        % acceptance rate to u
        lg_a_u = -phi_calc(u_prop,y_obs,obs_idx,sigma_n, L_curr,K)...
            + phi_calc(u_pcn(:,i),y_obs,obs_idx,sigma_n, L_curr,K);

        % acceptance
        if log(rand()) < lg_a_u
            u_pcn(:,i+1) = u_prop;
            accepted = accepted + 1;
        else
            % reject
            u_pcn(:,i+1) = u_pcn(:,i);
        end
        % update sigma_n
        lg_sigma_n_prop = log(sigma_n) + tuning_sig *randn();
        sigma_n_prop = exp(lg_sigma_n_prop);
        lg_sigma_n = log(sigma_n);
        X_curr = L_curr*u_pcn(:,i+1);
        X_all(:,i+1) = X_curr;
        % likelihood for sigma_n
        if command == "task2.2"
            y_curr = K * X_curr;
            resi = y_obs - y_curr;
            nll_curr = 0.5*sum(resi.^2)/(sigma_n^2) + ...
                (n_obs/2)*log(sigma_n^2);
            nll_prop = 0.5*sum(resi.^2)/(sigma_n_prop^2) + ...
                (n_obs/2)*log(sigma_n_prop^2);
        else
            nll_curr = (0.5*sum((y_obs - X_curr(obs_idx)).^2) /...
                (sigma_n^2)) + ((n_obs/2)*log(sigma_n^2));
            nll_prop = (0.5*sum((y_obs - X_curr(obs_idx)).^2) /...
                (sigma_n_prop^2)) + ((n_obs/2)*log(sigma_n_prop^2));
        end
        % evaluate e.g accept reject
        lp_sigma_prop = log_prior(sigma_n_prop,"sigma",prior);
        lp_sigma_curr = log_prior(sigma_n,"sigma",prior);
        lp_a_sig_n = (-nll_prop) - (-nll_curr) +...
            lp_sigma_prop - lp_sigma_curr +...
            (lg_sigma_n_prop - lg_sigma_n);
        if log(rand()) < lp_a_sig_n
            sigma_n = sigma_n_prop;
            acc_sigma_n = acc_sigma_n + 1;
        end
        samples_sigma_n(i+1) = sigma_n;
    end
    % sample processing
    u_pcn_samp = u_pcn(:, (burnIn+1):end);
    sigma_n_samp = samples_sigma_n(:, (burnIn+1):end);
    nu_samp = samples_nu(:, (burnIn+1):end);
    X_pcn_samp = X_all(:, (burnIn+1):end);

    mu_pcn = mean(X_pcn_samp,2);
    mu_sigma_n = mean(sigma_n_samp,2);
    mu_nu = mean(nu_samp,2);
    
    std_pcn = std(X_pcn_samp,0,2);
    std_sigma_n = std(sigma_n_samp,0,2);
    std_nu = std(nu_samp, 0,2);
    
    acc_rate = (accepted / (nums-1))*100;
    acc_rate_sigma_n = (acc_sigma_n /(nums-1))*100;
    acc_rate_nu = (accepted_nu /(nums-1))*100;
    % create output
    keys  = ["results", "mean", "std",...
        "acceptance rate","sigma_samp",...
        "sigma_mean", "sigma_std",...
        "full_u", "full_sigma", "acc_rate_sigma",...
        "nu_samp","nu_mean", "nu_std", "full_nu",...
        "acc_rate_nu"];
    values = {X_pcn_samp, mu_pcn,...
        std_pcn, acc_rate,...
        sigma_n_samp, mu_sigma_n,...
        std_sigma_n, u_pcn,...
        samples_sigma_n, acc_rate_sigma_n,...
        nu_samp, mu_nu,std_nu,...
        samples_nu,acc_rate_nu};
    res = containers.Map(keys,values);
end

% to calculate phi
function phi = phi_calc(u, y_obs, obs_idx, sigma, L, K)
    X = L * u;
    if ~isempty(K)
        % inverse problem path
        y_pred = K * X;
        residual = y_obs - y_pred;
        phi = 0.5 * sum(residual.^2) / (sigma^2);
    else
        % normal path
        phi = 0.5 * sum((y_obs - X(obs_idx)).^2) /...
            (sigma^2);
    end
end

function lp = log_prior(x, which,type)
    % uniform distribution for sigma_l
    if which == "sigma"
        lo = 0.05;
        hi = 1.35;
        m = log(0.2);
        s = 0.5;
    else
        % nu
        lo = 0.1;
        hi = 5;
        m = 0.5;
        s = 0.25;
    end
    if type == "uni"
        if x >= lo && x <= hi
            lp = -log(hi-lo);
        else
            lp = -Inf;
        end
    else
        % log normal
        if x > 0
            lp = -log(x*s*sqrt(2*pi)) -...
                0.5*((log(x)-m)/s)^2;
        else 
            lp = -Inf;
        end
    end
end

function log_post = log_poste(u, C1, y_obs,...
    obs_idx, sigma_n, K)
    N = length(u);
    L = chol(C1 + 1e-8*eye(N), 'lower');
    phi = phi_calc(u, y_obs, obs_idx, sigma_n, L,K);
    u_prior = 0.5*(u'* u);
    log_det_C1 = 0.5*(2*sum(log(diag(L))));
    log_post = -phi - u_prior -log_det_C1;
end

function K = forward_mat(t, lim_a, lim_b)
    N = length(t);
    dt = t(2) - t(1);

    K = zeros(N,N);
    
    for i = 1:N
        for j = 1:N
            tau = t(i) - t(j);
            K(i,j) = boxcar_fn(lim_a,tau,lim_b)*dt;
        end
    end
end

function bxc = boxcar_fn(lim_a, val, lim_b)
    if lim_a <= val && val <= lim_b
        bxc = 1;
    else
        bxc = 0;
    end
end
