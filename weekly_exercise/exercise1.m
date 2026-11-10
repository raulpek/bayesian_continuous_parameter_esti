clc
clearvars
close all

% Task 1

% params
h = 0.01;
i = linspace(0,100, 101);
t1 = (i*h)';
N = length(t1);
% 1) D=1 case
% Matern covariance
sigma_f = 1.0; % signal variance
sigma_l = 0.5; % length-scale
nu = 1.5;
C1 = matern_own(t1,nu,sigma_f, sigma_l);
% Cholesky factorization
[U_up, ~] = chol(C1 + 1e-8*eye(size(C1)));
U1 = U_up'; % lower triangular
W1 = randn(N,1);
X1 = U1 * W1; % realization X_omega

figure('Name','1D Matern Process realization');
plot(t1,X1, 'Color','green', 'LineStyle','-',...
    'LineWidth',1.5);
grid on;
xlabel('t');
ylabel('X(t)');
title('1D Gaussian Field realization, matern 1.5');
% 2) D = 2 case
[T1,T2] = meshgrid(0:h:1, 0:h:1);
grid_size = size(T1);
% flatten to N*2
t2_coords = [T1(:), T2(:)];
N2 = size(t2_coords,1);

C2 = matern_own(t2_coords, nu,...
    sigma_f, sigma_l);

[U2_up, ~] = chol(C2 + 1e-8*eye(size(C2)));
U2 = U2_up'; % lower triangular
W2 = randn(N2,1);
X2_flat = U2 * W2; % realization X_omega
% reshape
X2 = reshape(X2_flat, grid_size);

figure('Name','2D Matern Process realization');
surf(T1,T2,X2, 'EdgeColor','none');
colormap jet;
colorbar;
view(2);
axis tight;
xlabel('t');
ylabel('X(t)');
title('2D Gaussian Field realization, matern 1.5');
%%  Task 2
clc
clearvars
close
% params
h = 0.01;
t1 = (0:h:1)';
N = length(t1);
% Matern covariance
sigma_f = 1.0; % signal variance
sigma_l = 0.5; % length-scale
nu = 1.5;
sigma_n = 0.15; % measurement noise
C1 = matern_own(t1,nu,sigma_f, sigma_l);
% Cholesky factorization
[U_up, ~] = chol(C1 + 1e-8*eye(size(C1)));
U1 = U_up'; % lower triangular
% Ground truth
rng(42);
W_omega = randn(N,1);
X_omega = U1*W_omega;
% Noisy measurements
obs_idx = [15, 30, 45, 60 ,75];
t_obs = t1(obs_idx);
y_obs = X_omega(obs_idx) + ...
    sigma_n *...
    randn(length(obs_idx),1);
num_obs = length(y_obs);
% closed form GPR
K_obs = C1(obs_idx,obs_idx) +...
    (sigma_n^2)*eye(num_obs);
K_c = C1(:,obs_idx); % cross-variance
% posterior covariance and mean
L = K_obs \ K_c';
mu_post = L'*y_obs; % post mean
cov_post = C1 - K_c* L; % post covar
std_post = sqrt(max(0, diag(cov_post)));
% MCMC ... MH vs pCN
N_mcmc = 10000;
% _pcn_
u_pcn = zeros(N,N_mcmc);
u_pcn(:,1) = zeros(N,1);
%calc_phi = phi_calc(u_pcn(:,1),y_obs,...
%    obs_idx,sigma_n);
phi_curr = phi_calc(u_pcn(:,1),y_obs,...
    obs_idx,sigma_n);
acc_pcn = 0;
burn_in = 500;
u_pcn_ready = pcn_algo(u_pcn,U1,N, phi_curr,...
    y_obs, obs_idx, sigma_n);
mu_pcn = u_pcn_ready("mean");
std_pcn = u_pcn_ready("std");
u_mh_ready = MH_algo(N, N_mcmc,...
    C1, y_obs, obs_idx, sigma_n, burn_in);
% Plotting
figure('Name','GPR and MCMC state estimation');
% plot 1 closed-form gpr
subplot(3,1,1);
hold on;
fill([t1;flipud(t1)],...
    [mu_post+1.96*std_post;...
    flipud(mu_post-1.96*std_post)],...
    [0.85 0.85 1.0],'EdgeColor',...
    'none','DisplayName','p95 confidence');
plot(t1,X_omega, 'Color',...
    'black', 'LineWidth',1.5,...
    'DisplayName','True X_omega');
plot(t1, mu_post, 'Color','b',...
    'LineStyle','-','LineWidth',2,...
    'DisplayName','Analytical conditional mean');
plot(t_obs,y_obs,'Color','r',...
    'Marker','o','MarkerSize',8,...
    'MarkerFaceColor','r',...
    'DisplayName','y_e');
grid on;
ylabel('X(t)');
title('Closed-form GPR');
legend('Location','northeastoutside');
% plot 2 pcn mcmc
subplot(3,1,2);
hold on;
fill([t1;flipud(t1)],...
    [mu_pcn+1.96*std_pcn;...
    flipud(mu_pcn-1.96*std_pcn)],...
    [1.0 0.85 0.85],'EdgeColor',...
    'none','DisplayName','p95 confidence');
plot(t1,X_omega, 'Color',...
    'black', 'LineWidth',1.5,...
    'DisplayName','True X_omega');
plot(t1, mu_pcn, 'Color','b',...
    'LineStyle','-','LineWidth',2,...
    'DisplayName','pCN posterior mean');
plot(t_obs,y_obs,'Color','r',...
    'Marker','o','MarkerSize',8,...
    'MarkerFaceColor','r',...
    'DisplayName','y_e');
grid on;
ylabel('X(t)');
title('pCN MCMC');
legend('Location','northeastoutside');
% mh plot
u_mh_m = u_mh_ready("results");
u_mh_res = u_mh_m{1};
mu_mh = mean(u_mh_res,2);
std_mh = std(u_mh_res,0, 2);
subplot(3,1,3);
hold on;
fill([t1;flipud(t1)],...
    [mu_mh+1.96*std_mh;...
    flipud(mu_mh-1.96*std_mh)],...
    [1.0 0.85 0.85],'EdgeColor',...
    'none','DisplayName','p95 confidence');
plot(t1,X_omega, 'Color',...
    'black', 'LineWidth',1.5,...
    'DisplayName','True X_omega');
plot(t1, mu_mh, 'Color','b',...
    'LineStyle','-','LineWidth',2,...
    'DisplayName','pCN posterior mean');
plot(t_obs,y_obs,'Color','r',...
    'Marker','o','MarkerSize',8,...
    'MarkerFaceColor','r',...
    'DisplayName','y_e');
grid on;
ylabel('X(t)');
title('Metropolis Hastings');
legend('Location','northeastoutside');
% acceptance rates
u_pcn_acr = u_pcn_ready("acceptance rate"); % 34 percent
u_mh_acc = u_mh_ready("acceptance rate");
u_mh_acr = u_mh_acc{1}; % 0 percent

%% Functions

% Matern Covariance
function K = matern_own(X, nu, sigma_f, sigma_l)
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
function res = pcn_algo(u_pcn, U,...
    N, phi_curr, y_obs, obs_idx, sigma_n)
    nums = size(u_pcn,2); % n_mcmc
    accepted = 0;
    beta = 0.15;
    burnIn = 500;

    for i = 1:nums-1
        % sample e ~ N(0,C)
        e = U * randn(N,1);
        % proposal v
        v = sqrt(1-beta^2)*u_pcn(:,i) + beta*e;
        % acceptance probability
        phi_v = phi_calc(v, y_obs,...
            obs_idx, sigma_n);
        a = min(1, exp(phi_curr - phi_v));
        % threshold
        g = rand(); % uniform sample
        if g <= a
            u_pcn(:, i+1) = v; % new state
            phi_curr = phi_v; % new potential
            if i > burnIn
                accepted = accepted + 1;
            end
        else
            u_pcn(:, i+1) = u_pcn(:,1);
        end
    end
    % sample processing
    u_pcn_samp = u_pcn(:, (burnIn+1):end);
    mu_pcn = mean(u_pcn_samp,2);
    std_pcn = std(u_pcn_samp,0,2);
    acc_rate = (accepted / (nums-burnIn))*100;
    % create output
    keys  = ["results", "mean", "std",...
        "acceptance rate"];
    values = {u_pcn_samp, mu_pcn,...
        std_pcn, acc_rate};
    res = containers.Map(keys,values);
end

function MH_res = MH_algo(N1, n_mcmc,...
    C1, y_obs, obs_idx, sigma_n, burn_in)
    step = 0.03;
    u_mh = zeros(N1,n_mcmc);
    u_mh(:,1) = zeros(N1,1);
    accepted = 0;

    invC1 = inv(C1 + 1e-8*eye(N1));
    log_post_curr = log_poste(u_mh(:,1), invC1, y_obs, obs_idx, sigma_n);
    for i = 1:(n_mcmc-1)
        v = u_mh(:,i) + step * randn(N1,1);
        log_post_prop = log_poste(v, invC1, y_obs,...
            obs_idx, sigma_n) -...
            log_poste(u_mh(:,1),invC1,...
            y_obs, obs_idx, sigma_n);
        log_a = log_post_prop - log_post_curr;
        % acceptance
        g = rand();
        if log(g) <= log_a
            u_mh(:,i+1) = v;
            log_post_curr = log_post_prop;
            if i > burn_in
               accepted = accepted + 1;
            end
        else
            u_mh(:, i+1) = u_mh(:, i);
        end
    end
    % sample processing
    % create output
    acc_rate = (accepted / (n_mcmc-burn_in))*100;
    % create output
    keys  = ["results",...
        "acceptance rate"];
    values = {u_mh, acc_rate};
    MH_res = dictionary(keys,values);
end
% to calculate phi
function phi = phi_calc(u, y_obs, obs_idx, sigma)
    phi = 0.5 * sum((y_obs - u(obs_idx)).^2) /...
        (sigma^2);
end

function log_post = log_poste(u, invC1, y_obs,...
    obs_idx, sigma_n)
    log_post = -phi_calc(u, y_obs,obs_idx,sigma_n) -0.5 *...
        (u'*invC1 *u);
end
