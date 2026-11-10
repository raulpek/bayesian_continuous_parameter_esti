%% Task1
clc
clearvars
close all

% grid 
h = 0.01;
t1 = (0:100)'*h;
N = length(t1);
% ground truth
R_dist_N = abs(t1-t1');
% parameters
nu = 1.5;
sigma_l_true = 0.40;
sigma_f = 1.0;
sigma_n = 0.10;
rng(42);
[X, C] = matern_own(R_dist_N,sigma_l_true,nu,sigma_f);
%C_inv = inv(C);
C_inv = C \ eye(N);
X_true = X * randn(N,1); % a realization
% kernel
theta_true = 0.5;
K_true = box_car_fn(h, theta_true, theta_true/2,N, 0);
y = K_true*X_true + sigma_n*randn(N,1);
% point-estimates
bounds = [0.02, 0.30];
J = @(z) 0.5*sum((y - box_car_fn(h, z(end),z(end)/2,...
    N,0) * z(1:N)).^2) / sigma_n^2 ...
    + 0.5*z(1:N)'* (C_inv\z(1:N));
z0 = [zeros(N,1); 0.2];
low_b = [-inf(N,1); bounds(1)];
up_b = [inf(N,1); bounds(2)];

opt = optimoptions('fmincon','Algorithm','interior-point',...
    'HessianApproximation','lbfgs','FiniteDifferenceStepSize',0.02,...
    'MaxIterations',500,'Display','off');
zmap = fmincon(J, z0, [], [], [], [], low_b, up_b,[], opt);
xmap = zmap(1:N);
theta_map = zmap(end);

% checks
th_grid = bounds(1):0.02:bounds(2);
J_grid = zeros(size(th_grid));
for k = 1:length(th_grid)
    kk = box_car_fn(h, th_grid(k), th_grid(k)/2, N, 0);
    xk = (kk' * kk/sigma_n^2 + C_inv) \ (kk' *y / sigma_n^2);
    J_grid(k) = J([xk; th_grid(k)]);
end
[~, kbest] = min(J_grid);
% gibbs mh
burnIn = 2000;
nMcmc = 10000;
step = 0.06;
theta_ini = 0.2;
K_curr = box_car_fn(h, theta_ini, theta_ini/2, N, 0);

result = gibbs_mh(N, nMcmc,burnIn, step, K_curr, sigma_n, C_inv,...
    y, theta_ini, bounds, h, "task1",[0,0,0, 0]);
% plotting
plot_params = {theta_true, X_true, xmap, t1, y}; % 5
plotting(result, plot_params, "task1");
%% Task 2
clc
clearvars
close all

% grid 
h = 0.01;
t1 = (0:100)'*h;
N = length(t1);
% ground truth
R_dist_N = abs(t1-t1');
% parameters
M = 20;
nu = 1.5;
sigma_l_true = 0.40;
sigma_f = 1.0;
sigma_n = 0.10;
theta_true = 0.5;
rng(42);
[X, C] = matern_own(R_dist_N,sigma_l_true,nu,sigma_f);
X_true = X*randn(N,M); % M num of realizations
% kernel
K_true = box_car_fn(h, theta_true, theta_true/2,N, 0);
k_true = K_true(:,1);
y = K_true*X_true + sigma_n*randn(N,M);
y_all = y(:);
% kx as linear function
Aall = zeros(M*N,N);
[r,c] = ndgrid(1:N, 1:N);
lags_i = abs(r - c) +1;
for i = 1:M
    A = zeros(N,N);
    for ri = 1:N
        A(ri, :) = accumarray(lags_i(ri, :)', X_true(:,i), [N 1])';
    end
    Aall((i-1)*N+1: i*N, :) = A;
end

% gaussian prior over the lags
lags_k= (0:N-1)'*h;
R_dist_k = abs(lags_k -lags_k');
sigma_k = 0.10;
l_k = 0.05;
[L_k, Cov_k] = matern_own(R_dist_k, l_k, nu, sigma_k);
C_inv = L_k' \ (L_k \ eye(N)); % inv prior cov
% point-estimate lgbfs
J = @(k) loss_n_grad(k, Aall, y_all, L_k, sigma_n);
opt = optimoptions('fmincon','Algorithm','interior-point',...
    'HessianApproximation','lbfgs',...
    'MaxFunctionEvaluations',1e5,'Display','off');
[kmap, ~] = fmincon(J, zeros(N,1), [], [], [], [], [], [],[], opt);
% confirm that posterior is gaussian
P = (Aall' * Aall)/ sigma_n^2 + C_inv;
b = (Aall' * y_all) / sigma_n^2;
k_prec = P \ b;
% gibbs MH
k = zeros(N,1);
n_iter = 6000;
burnIn = 1000;
result = gibbs_mh(N, n_iter,burnIn, 0, 0, 0, 0,...
    0, 0, 0, 0, "task2", {n_iter,kmap,P, b});

plot_params = {k_true, lags_k, kmap, N}; % 4 params
plotting(result, plot_params, "task2");
%% Functions

% box car function
function K = box_car_fn(h, w, half_w, N, extend)
    m = round(half_w/ h);
    if isscalar(N)
        K = zeros(N,N);
        for i = 1:N
            for j = 1:N
                if abs(i-j) <= m
                    K(i,j) = 1/w *h;
                end
            end
        end
    else
        % extended domain
        K = zeros(N(1),N(2));
        for i = 1:N(1)
            for j = 1:N(2)
                if abs((i+extend)-j) <= m
                    K(i,j) = 1/w *h;
                end
            end
        end

    end
end

% Matern Covariance
function [X,C] = matern_own(R_dist, sigma_l, nu, sigma_f)
    N = size(R_dist,1);
    C = matern_cov(R_dist, sigma_l, nu, sigma_f);
    X = chol(C + 1e-6*eye(N), 'lower');
end

function K = matern_cov(R_dist, sigma_l, nu, sigma_f)
    if nu == 1.5
        arg = (sqrt(3)/sigma_l)*R_dist;
        K = sigma_f^2 * (1 + arg) .* exp(-arg);
    else
        r = R_dist;
        r(r==0) = eps;
        arg = (sqrt(2*nu)*r/sigma_l);
        K = sigma_f^2 *(2^(1-nu)/gamma(nu))*(arg.^nu) .*...
            besselk(nu,arg);
        K(logical(eye(size(K)))) = sigma_f^2;
    end
end

function [val, grad]= loss_n_grad(k, Aall, y_all, L_k, sigma_n)
    resi = y_all - Aall *k;
    v = L_k \ k;
    C_inv_k = L_k' \ v;
    val = 0.5*sum(resi.^2) / (sigma_n^2) + 0.5...
        *(k' * C_inv_k);
    grad = -(Aall' * resi) / (sigma_n^2) + C_inv_k;
end


% Gibbs MH
function res = gibbs_mh(N, n_iter,burnIn, step, K_curr, sigma_n, Cinv,...
    y, theta, bounds, h, task, nKP_b)
   theta_chain = zeros(n_iter,1);
   x_chain = zeros(N,n_iter);
   acc_theta = 0;
   if task == "task2"
       n = nKP_b{1};
       k = nKP_b{2};
       P = nKP_b{3};
       b = nKP_b{4};
       k_chain = zeros(N,n);
       for it = 1:n_iter
           for j = 1:N
               mu_j = (b(j) - P(j,:)*k + P(j,j)*...
                   k(j)) / P(j,j);
               k(j) = mu_j + randn/sqrt(P(j,j));
           end
           k_chain(:,it) = k;
       end
       k_samp = k_chain(:, burnIn+1:end);
       n = size(k_samp,2);
       res.k_samp = k_samp;
       res.n = n;
   else
       for it = 1:n_iter
           % gibbs step
           A = K_curr' * K_curr/sigma_n^2 + Cinv;
           A = 0.5*(A+A');
           R = chol(A);
           m = A \ (K_curr' *y / sigma_n^2);
           x = m + R \ randn(N,1);
           r_curr = y - K_curr*x;
           ss_curr = r_curr' * r_curr;
           % MH
           theta_prop = theta + step*randn;
           if theta_prop > bounds(1) && theta_prop < bounds(2)
               K_prop = box_car_fn(h, theta_prop, theta_prop/2,N,0);
               r_prop = y - K_prop*x;
               ss_prop = r_prop' *r_prop;
               logAlpha = -0.5/sigma_n^2 * (ss_prop - ss_curr);
               % acceptance
               if log(rand) < logAlpha
                   theta = theta_prop;
                   K_curr = K_prop;
                   acc_theta = acc_theta + 1;
               end
           end
           x_chain(:,it) = x;
           theta_chain(it) = theta;
       end
       res = struct();
       res.theta_chain_full = theta_chain;
       res.x_chain_full = x_chain;
       res.acceptance_rate = acc_theta / n_iter;
       res.theta_samp = theta_chain(burnIn+1:end);
       res.x_samp = x_chain(:, burnIn+1:end);
       res.n = length(res.theta_samp);
       fprintf('Acceptance rate for theta: %.2f\n', res.acceptance_rate);
   end
end

function plotting(res,params, task)
    max_lag = 100;
    if task == "task1"
        % extract params
        theta_true = params{1};
        X_true = params{2};
        xmap = params{3};
        t1 = params{4};
        y = params{5};
        % compute metrics
        theta_samp = res.theta_samp(:);
        x_samp = res.x_samp;
        if size(x_samp,1) ~= length(t1) && size(x_samp,2) == length(t1)
            x_samp = x_samp';
        end
        [n_samp,n_params] = size(theta_samp);
        acf = zeros(max_lag+1,n_params);
        for j = 1:n_params
            acf(:,j) = simple_acf(theta_samp(:,j),max_lag);
        end
        % acf = simple_acf(res.k_samp,max_lag);
        % ess
        ess_all = zeros(1, n_params);
        for j = 1:n_params
            k_nega = find(acf(2:end, j) < 0, 1);
            if isempty(k_nega)
                k_nega = max_lag;
            end
            ess_all(j) = n_samp / (1+2*sum(acf(2:k_nega+1, j)));
        end
        ESS = mean(ess_all);
        % 95per intervals
        theta_mean = mean(theta_samp,1);
        theta_ci_low = prctile(theta_samp,2.5, 1);
        theta_ci_hi = prctile(theta_samp,97.5,1);
        x_mean = mean(x_samp,2);
        x_ci_low = prctile(x_samp,2.5,2);
        x_ci_hi = prctile(x_samp,97.5,2);
        rmse_postm = sqrt(mean((x_mean(:) - X_true(:)).^2));
        rmse_x_lbfs = sqrt(mean((xmap(:) - X_true(:)).^2));
        fprintf("ESS of theta: %.3f out of %d samples.\n", ESS,n_samp);
        fprintf("RMSE_x posterior mean: %.3f\n", rmse_postm);
        fprintf("RMSE_x l-bfs: %.3f\n", rmse_x_lbfs);
        % plots
        figure;
        subplot(2,1,1);
        plot(res.theta_chain_full);
        hold on;
        yline(theta_true,'r');
        xlabel("Iteration");
        ylabel("theta");
        title("Chain of theta");
        subplot(2,1,2);
        plot(res.x_chain_full(50,:));
        xlabel("Iteration");
        ylabel("x(t_50)");
        title("Chain of one component");
        % acf plot
        figure;
        stem(0:max_lag, acf(:,1));
        xlabel("lag");
        ylabel("ACF");
        title("ACF of theta chain");
        % marginal posterior
        figure;
        histogram(theta_samp, 'Normalization','pdf');
        xline(theta_true, 'r', 'LineWidth',1.8);
        xline(theta_ci_low,'k--');
        xline(theta_ci_hi,'k--');
        title("Marginal posterior of theta");
        xlabel("theta");
        ylabel("Density");
        grid on;
        % 95per credible interval
        xfill = [t1(:); flipud(t1(:))];
        yfill = [x_ci_low(:); flipud(x_ci_hi(:))];
        figure;
        fill(xfill,yfill,...
            [0.8 0.8 0.8], 'EdgeColor','none');
        hold on;
        plot(t1, X_true,'k', 'LineWidth',1.5);
        plot(t1, x_mean,'b', 'LineWidth',1.5);
        plot(t1, xmap,'--', 'Color',[1 0.5 0],'LineWidth',1.5);
        plot(t1,y,'g.');
        legend("95% CI", "true x", "posterior mean", "L-BFGS", "y");
        xlabel("t");
        title("Estimate of x");
    else
        k_true = params{1};
        lags = params{2};
        kmap = params{3};
        N = params{4};
        % compute metrics
        k_samp = res.k_samp;
        if size(k_samp,1) == N && size(k_samp,2) ~= N
            k_samp = k_samp';
        end
        [n_samp,n_params] = size(k_samp);
        acf = zeros(max_lag+1,n_params);
        for j = 1:n_params
            acf(:,j) = simple_acf(k_samp(:,j),max_lag);
        end
        % acf = simple_acf(res.k_samp,max_lag);
        % ess
        ess_all = zeros(1, n_params);
        for j = 1:n_params
            k_nega = find(acf(2:end, j) < 0, 1);
            if isempty(k_nega)
                k_nega = max_lag;
            end
            ess_all(j) = n_samp / (1+2*sum(acf(2:k_nega+1, j)));
        end
        ESS = mean(ess_all);
        fprintf("ESS of theta: %.3f out of %d samples.\n", ESS,n_samp);
        % 95per intervals
        k_mean = mean(res.k_samp,2);
        k_ci_low = prctile(res.k_samp,2.5, 2);
        k_ci_hi = prctile(res.k_samp,97.5,2);
        % 95per credible interval
        figure;
        fill([lags(:); flipud(lags(:))].', [k_ci_low(:);flipud(k_ci_hi(:))].',...
            [0.8 0.8 0.8], 'EdgeColor','none');
        hold on;
        plot(lags, k_true,'k', 'LineWidth',1.5);
        plot(lags, k_mean(:),'b', 'LineWidth',1.5);
        plot(lags, kmap,'--', 'Color',[1 0.5 0],'LineWidth',1.5);
        legend("95% CI", "true k", "MAP (l-bfgs)", "Gibbs mean");
        xlabel("lag");
        ylabel("kernel weight");
        title("Kernel learning");
        % k at lag and acf
        idx = [1, 6, 11];
        valid_idx = idx(idx <= n_params);
        l_idx = length(valid_idx);
        figure;
        for q = 1:length(idx)
            q_param = valid_idx(q);
            subplot(l_idx, 1,q);
            plot(k_samp(:, q_param));
            ylabel(["k at lag ", num2str(idx(q))]);
            if q == 1
                title("Gibbs chains");
            end
        end
        xlabel("iteration");

        figure;
        for q = 1:length(idx)
            q_param = valid_idx(q);
            subplot(l_idx, 1,q);
            stem(0:max_lag, acf(:, q_param), 'Marker','none');
            ylabel(["ACF,lag ", num2str(lags(idx(q)))]);
            if q == 1
                title("ACF");
            end
        end
        xlabel("lag iteration");

    end
end

function a = simple_acf(x, maxlag)
    x = x(:) - mean(x);
    c = zeros(maxlag+1,1);
    for k = 0:maxlag
        c(k+1) = sum(x(1:end-k) .* x(1+k:end));
    end
    a = c / c(1);
end
