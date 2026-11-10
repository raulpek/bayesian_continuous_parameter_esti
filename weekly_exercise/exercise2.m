clc
clearvars
close all

% TASK 1
% params
h = 0.01;
i = linspace(0,100, 101);
t1 = (i*h)';
N = length(t1);
% Matern covariance 1D
sigma_f = 1.0; % signal variance
sigma_l = 0.45; % length-scale, l true
sigma_n = 0.1; % noise
nu_true = 1.75; % ν = 1.75.
nu_alt = 1:0.15:5;
l_alt = 0.05:0.05:1.35;
num_models = length(nu_alt); % number of models
C_true = matern_own(t1,[sigma_l,nu_true],sigma_f);
% Cholesky factorization
[U_up, ~] = chol(C_true + 1e-8*eye(size(C_true)));
U1 = U_up'; % lower triangular
% Ground truth
rng(42);
W_omega = randn(N,1);
X_omega = U1*W_omega; % with v = 1.75
% and plot it
figure;
plot(t1, X_omega,'LineStyle','-');
xlabel("t")
ylabel("X(t)");
title("One realization of Matérn 1.75");
grid on;

% Noisy measurements
obs_idx = [15, 30, 45, 60 ,75];
t_obs = t1(obs_idx);

num_obs = length(obs_idx);
trials = 300;
rmse_trials_nu = zeros(num_models, trials);
rmse_trials_sig = zeros(num_models, trials);
for trial = 1:trials
    % fresh noise measurement for each trial
    y_obs = X_omega(obs_idx) + ...
        sigma_n *...
        randn(length(obs_idx),1);
    for v_test = 1:num_models
        % DIFFERENT NU, fixed sigma_l
        v_curr = nu_alt(v_test);
        C1 = matern_own(t1, [sigma_l, v_curr],sigma_f);
        K_obs = C1(obs_idx,obs_idx) +...
            (sigma_n^2)*eye(num_obs);
        K_c = C1(:,obs_idx);
        % log marginal likelihood
        L_K = chol(K_obs,'lower');
        alpha = L_K' \ (L_K \ y_obs);
        mu_post = K_c * alpha;
        % rmse and neg log marginal likelihood
        rmse_trials_nu(v_test, trial) = sqrt(mean((X_omega(:) - mu_post(:)).^2));
        % DIFFERENT SIGMA_L, fixed nu
        sigma_l_curr = l_alt(v_test);
        C2 = matern_own(t1, [sigma_l_curr,nu_true], sigma_f);
        K_obs_sig = C2(obs_idx, obs_idx) + (sigma_n^2) * eye(num_obs);
        K_c_sig = C2(:, obs_idx);
        L_K_sig = chol(K_obs_sig, 'lower');
        alpha_sig = L_K_sig' \ (L_K_sig \ y_obs);
        mu_post_sig = K_c_sig * alpha_sig;
        rmse_trials_sig(v_test, trial) = sqrt(mean((X_omega(:) - mu_post_sig(:)).^2));

    end
end

% try the different v
mean_rmse_nu = mean(rmse_trials_nu,2);
mean_rmse_sig = mean(rmse_trials_sig,2);
% Plotting rmse nu
figure;
plot(nu_alt, mean_rmse_nu,'LineStyle','--');
xlabel("ν")
ylabel("RMSE");
title("RMSE per v");
grid on;
% Plotting rmse sigma_l
figure;
plot(l_alt, mean_rmse_sig,'LineStyle','--');
xlabel("sigma_l")
ylabel("RMSE");
title("RMSE per sigma_l");
grid on;

% TASK 2

% hierarchical model
n_MCMC = 8000;
u_pcn = zeros(N, n_MCMC);
u_pcn(:,1) = zeros(N,1);
hyp_param_pcn = zeros(2,n_MCMC);

samples_theta = zeros(n_MCMC,2); % [sigma_l, nu]
samples_X = zeros(n_MCMC, length(t1));

ini_sig = 0.15; % initial guess
ini_nu = 1.2;% initial guess
theta(:,1) = [ini_sig, ini_nu];
C2 = matern_own(t1,theta(:,1), sigma_f);
L_curr = chol(C2 + 1e-8*eye(N), 'lower');

% call the algoritm
results = pcn_algo_joint(u_pcn,C2, theta,...
    L_curr, sigma_f, t1, y_obs,N,...
    obs_idx,sigma_n);

% plotting
u_mean = results("mean");
u_std = results("std");
figure;
subplot(2,2, [1,2]);
fill([t1;flipud(t1)],...
    [u_mean - 2*u_std;...
    flipud(u_mean+2*u_std)],...
    [0.85 0.85 1.0],'EdgeColor',...
    'none','DisplayName','p95 confidence');
hold on;
plot(t1,X_omega, 'Color',...
    'black', 'LineWidth',1.5,...
    'DisplayName','True X_omega');
plot(t1, u_mean, 'Color','b',...
    'LineStyle','-','LineWidth',2,...
    'DisplayName','Sample Posterior mean');
plot(t_obs,y_obs,'Color','r',...
    'Marker','o','MarkerSize',8,...
    'MarkerFaceColor','r',...
    'DisplayName','y_e');
grid on;
xlabel('t');
ylabel('X(t)');
title('Joint-estimation of GP state','FontSize',8);
legend('Location','northeastoutside');

% sigma_l
subplot(2,2,3);
full_theta = results("full_theta");
theta_mean = results("theta_mean");
plot(1:1000,full_theta(1,1:1000),'Color','g');
hold on;
plot(1001:n_MCMC, full_theta(1,1001:end),'b');
yline(sigma_l, 'r--','LineWidth',1.8);
xlabel('Iteration');
ylabel("sigma_l");
title(sprintf('Lenght-scale sigma_l (mean=%.2f)', theta_mean(1)),'FontSize',8);
grid on;

% nu
subplot(2,2,4);
plot(1:1000,full_theta(2,1:1000),'Color','g');
hold on;
plot(1001:n_MCMC, full_theta(2,1001:end),'b');
yline(nu_true, 'r--','LineWidth',1.8);
xlabel('Iteration');
ylabel("v");
title(sprintf('Smooth factor v (mean=%.2f)', theta_mean(2)), 'FontSize',8);
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

% PCN
function res = pcn_algo_joint(u_pcn,C_curr,theta,...
    L_curr,sigma_f,t1,y_obs,N,...
    obs_idx, sigma_n)
    nums = size(u_pcn,2); % n_mcmc
    accepted = 0;
    acc_theta = 0;
    beta = 0.15;
    burnIn = 1000;
    for i = 1:nums-1
        % current vals
        l_curr = theta(1,i);
        nu_curr = theta(2,i);
        xi = L_curr * randn(N,1);
        % proposals of u
        u_prop = sqrt(1-beta^2)*u_pcn(:,i) + beta*xi;
        % acceptance rate to u
        lg_a_u = -phi_calc(u_prop,y_obs,obs_idx,sigma_n)...
            + phi_calc(u_pcn(:,i),y_obs,obs_idx,sigma_n);
        if log(rand()) < lg_a_u
            u_pcn(:,i+1) = u_prop;
            accepted = accepted + 1;
        else
            u_pcn(:,i+1) = u_pcn(:,i);
        end
        % update theta
        l_prop = l_curr + 0.02*randn();
        nu_prop = nu_curr+ 0.04*randn();
        % evaluate
        lp_theta_prop = log_prior([l_prop,...
            nu_prop]);
        if isinf(lp_theta_prop)
            % theta(:,i+1) = theta(:,i);
            theta(1,i+1) = l_curr;
            theta(2,i+1) = nu_curr;
        else
            % new theta
            C_prop = matern_own(t1,[l_prop,nu_prop],sigma_f);
            lp_post_prop = log_poste(u_pcn(:,i+1), C_prop,y_obs,...
                obs_idx, sigma_n) + lp_theta_prop;
            lp_post_curr = log_poste(u_pcn(:,i+1), C_curr,y_obs,...
                obs_idx, sigma_n) + log_prior([l_curr, nu_curr]);

            log_a_theta = lp_post_prop - lp_post_curr;
            % acceptance of new params
            if log(rand()) < log_a_theta
                theta(1,i+1) = l_prop;
                theta(2,i+1) = nu_prop;
                C_curr = C_prop;
                L_curr = chol(C_curr + 1e-8*eye(N),'lower');
                acc_theta = acc_theta +1;
            else
                theta(1,i+1) = l_curr;
                theta(2,i+1) = nu_curr;
            end
        end
    end
    % sample processing
    u_pcn_samp = u_pcn(:, (burnIn+1):end);
    theta_samp = theta(:, (burnIn+1):end);

    mu_pcn = mean(u_pcn_samp,2);
    mu_theta = mean(theta_samp,2);

    std_pcn = std(u_pcn_samp,0,2);
    std_theta = std(theta_samp,0,2);

    acc_rate = (accepted / (nums-1))*100;
    acc_rate_theta = (acc_theta /(nums-1))*100;
    % create output
    keys  = ["results", "mean", "std",...
        "acceptance rate","theta_samp",...
        "theta_mean", "theta_std",...
        "full_u", "full_theta", "acc_rate_theta"];
    values = {u_pcn_samp, mu_pcn,...
        std_pcn, acc_rate,...
        theta_samp, mu_theta,...
        std_theta, u_pcn,...
        theta, acc_rate_theta};
    res = containers.Map(keys,values);
end

% to calculate phi
function phi = phi_calc(u, y_obs, obs_idx, sigma)
    phi = 0.5 * sum((y_obs - u(obs_idx)).^2) /...
        (sigma^2);
end

function log_prio = log_prior(theta)
    sigma_l = theta(1);
    nu = theta(2);
    % uniform distribution for sigma_l
    if 0.05 <= sigma_l && sigma_l <= 1.35
        l_sigma = log(1/1.30);
    else
        l_sigma = -inf;
    end
    % log normal for nu
    if nu > 0
        l_nu = log(pdf('LogNormal',nu,0.5,0.25));
    else
        l_nu = -inf;
    end
    log_prio = l_sigma + l_nu;
end

function log_post = log_poste(u, C1, y_obs,...
    obs_idx, sigma_n)
    N = length(u);
    phi = phi_calc(u, y_obs, obs_idx, sigma_n);
    u_prior = 0.5*(u' *(C1 \ u));
    L = chol(C1 + 1e-8*eye(N), 'lower');
    log_det_C1 = 0.5*(2*sum(log(diag(L))));
    log_post = -phi - u_prior -log_det_C1;
end
